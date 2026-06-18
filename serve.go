// serve.go
//
// things for serving oll itself as an MCP server / tool

package main

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"math/rand/v2"
	"net/http"
	"net/url"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"slices"
	"strings"
	"syscall"
	"time"

	"github.com/gabriel-vasile/mimetype"
	"github.com/google/jsonschema-go/jsonschema"
	"github.com/meinside/version-go"
	"github.com/modelcontextprotocol/go-sdk/mcp"
	"github.com/ollama/ollama/api"
	mdl "github.com/ollama/ollama/types/model"
)

const (
	mcpServerName   = `oll/mcp-server`
	mcpToolNameSelf = `oll/mcp-self`

	mcpFunctionTimeoutSeconds = 3 * 60

	commandTimeoutSeconds = 30
)

// funcArg extracts a typed argument from the given args map.
//
// Returns (nil, nil) if the key is absent, an error if present but of the wrong type.
func funcArg[T any](args map[string]any, key string) (*T, error) {
	v, ok := args[key]
	if !ok || v == nil {
		return nil, nil
	}
	if typed, ok := v.(T); ok {
		return &typed, nil
	}
	return nil, fmt.Errorf("argument '%s' is not of expected type %T", key, *new(T))
}

// mcpTextResult returns a success CallToolResult with text content.
func mcpTextResult(text string) (*mcp.CallToolResult, error) {
	return &mcp.CallToolResult{
		Content: []mcp.Content{
			&mcp.TextContent{Text: text},
		},
	}, nil
}

// mcpJSONResult returns a success CallToolResult with both text and structured JSON.
func mcpJSONResult(marshalled []byte) (*mcp.CallToolResult, error) {
	return &mcp.CallToolResult{
		Content: []mcp.Content{
			&mcp.TextContent{Text: string(marshalled)},
		},
		StructuredContent: json.RawMessage(marshalled),
	}, nil
}

// mcpErrorResult returns an error CallToolResult with a formatted message.
func mcpErrorResult(format string, a ...any) (*mcp.CallToolResult, error) {
	return &mcp.CallToolResult{
		Content: []mcp.Content{
			&mcp.TextContent{Text: fmt.Sprintf(format, a...)},
		},
		IsError: true,
	}, nil
}

// buildSelfServer builds an MCP server exposing oll itself as tools.
func buildSelfServer(
	output *outputWriter,
	conf config,
	p params,
) (*mcp.Server, []*mcp.Tool) {
	server := mcp.NewServer(
		&mcp.Implementation{
			Name:    mcpServerName,
			Version: version.Build(version.OS | version.Architecture),
		},
		&mcp.ServerOptions{},
	)

	type toolAndHandler struct {
		tool    mcp.Tool
		handler mcp.ToolHandler
	}
	toolsAndHandlers := make([]toolAndHandler, 0)

	// oll_list_models (read only, idempotent)
	toolsAndHandlers = append(toolsAndHandlers, toolAndHandler{
		tool: mcp.Tool{
			Name:        `oll_list_models`,
			Description: `Use this function when you need to know which Ollama models are installed locally, for example before selecting a model for generation.`,
			InputSchema: &jsonschema.Schema{
				Type:     "object",
				ReadOnly: true,
			},
			Annotations: &mcp.ToolAnnotations{
				IdempotentHint: true,
				ReadOnlyHint:   true,
			},
		},
		handler: func(
			ctx context.Context,
			request *mcp.CallToolRequest,
		) (*mcp.CallToolResult, error) {
			client, err := newOllamaClient()
			if err != nil {
				return mcpErrorResult("Failed to initialize Ollama API client: %s", err)
			}
			models, err := client.List(ctx)
			if err != nil {
				return mcpErrorResult("Failed to list models: %s", err)
			}
			marshalled, err := json.Marshal(models)
			if err != nil {
				return mcpErrorResult("Failed to marshal models: %s", err)
			}
			return mcpJSONResult(marshalled)
		},
	})

	// oll_generate
	toolsAndHandlers = append(toolsAndHandlers, toolAndHandler{
		tool: mcp.Tool{
			Name: `oll_generate`,
			Description: `Use this function when you need to generate text or an image from a prompt, optionally using local files as context.

* NOTE:
- If an image is generated, it is saved to a local file; the absolute filepath is reported so the user can use it later.`,
			InputSchema: &jsonschema.Schema{
				Type: "object",
				Properties: map[string]*jsonschema.Schema{
					"prompt": {
						Title:       "prompt",
						Description: `The user's prompt for generation.`,
						Type:        "string",
					},
					"modality": {
						Title:       "modality",
						Description: `The modality of the generation. Must be one of 'text' or 'image'.`,
						Type:        "string",
						Enum:        []any{"text", "image"},
					},
					"filepaths": {
						Title:       "filepaths",
						Description: `Paths to local files to be processed along with the given 'prompt'. Relative paths are resolved against the current working directory.`,
						Type:        "array",
					},
					"model": {
						Title:       "model",
						Description: `The model to use for generation. If not specified, the default model for the modality will be used.`,
						Type:        "string",
					},
					"with_thinking": {
						Title:       "with_thinking",
						Description: `Whether to generate with thinking. Ignored unless 'modality' is 'text'. Default false.`,
						Type:        "boolean",
					},
					"convert_url": {
						Title:       "convert_url",
						Description: `Whether to convert URLs in the prompt into their fetched contents. Ignored unless 'modality' is 'text'. Default false.`,
						Type:        "boolean",
					},
					"negative_prompt": {
						Title:       "negative_prompt",
						Description: `Negative prompt for image generation. Ignored unless 'modality' is 'image'.`,
						Type:        "string",
					},
					"image_width": {
						Title:       "image_width",
						Description: `Width for image generation. Ignored unless 'modality' is 'image'.`,
						Type:        "integer",
					},
					"image_height": {
						Title:       "image_height",
						Description: `Height for image generation. Ignored unless 'modality' is 'image'.`,
						Type:        "integer",
					},
					"image_seed": {
						Title:       "image_seed",
						Description: `Seed for image generation. Ignored unless 'modality' is 'image'.`,
						Type:        "integer",
					},
				},
				Required: []string{"prompt", "modality"},
			},
		},
		handler: selfGenerateHandler(output, conf, p),
	})

	// oll_get_cwd (read only, idempotent)
	toolsAndHandlers = append(toolsAndHandlers, toolAndHandler{
		tool: mcp.Tool{
			Name: `oll_get_cwd`,
			Description: `Use this function when you need the current working directory (absolute path), for example before resolving or handling relative filepaths.

* NOTE:
- It is advised to call this function before performing any task which handles filepaths.`,
			InputSchema: &jsonschema.Schema{
				Type:     "object",
				ReadOnly: true,
			},
			Annotations: &mcp.ToolAnnotations{
				IdempotentHint: true,
				ReadOnlyHint:   true,
			},
		},
		handler: func(
			ctx context.Context,
			request *mcp.CallToolRequest,
		) (*mcp.CallToolResult, error) {
			cwd, err := os.Getwd()
			if err != nil {
				return mcpErrorResult("Failed to get current working directory: %s", err)
			}
			marshalled, err := json.Marshal(struct {
				Cwd string `json:"currentWorkingDirectory"`
			}{Cwd: cwd})
			if err != nil {
				return mcpErrorResult("Failed to marshal current working directory: %s", err)
			}
			return mcpJSONResult(marshalled)
		},
	})

	// oll_list_envvar_names (read only)
	toolsAndHandlers = append(toolsAndHandlers, toolAndHandler{
		tool: mcp.Tool{
			Name: `oll_list_envvar_names`,
			Description: `Use this function when you need to know which environment variables exist, for example before retrieving a specific one's value.

* NOTE:
- This function should be called prior to 'oll_get_envvar'.`,
			InputSchema: &jsonschema.Schema{
				Type:       "object",
				ReadOnly:   true,
				Properties: map[string]*jsonschema.Schema{},
			},
			Annotations: &mcp.ToolAnnotations{
				ReadOnlyHint: true,
			},
		},
		handler: func(
			ctx context.Context,
			request *mcp.CallToolRequest,
		) (*mcp.CallToolResult, error) {
			names := []string{}
			for _, env := range os.Environ() {
				if name, _, found := strings.Cut(env, "="); found {
					names = append(names, name)
				}
			}
			marshalled, err := json.Marshal(names)
			if err != nil {
				return mcpErrorResult("Failed to marshal environment variables' names: %s", err)
			}
			return mcpJSONResult(marshalled)
		},
	})

	// oll_get_envvar (read only, destructive)
	toolsAndHandlers = append(toolsAndHandlers, toolAndHandler{
		tool: mcp.Tool{
			Name: `oll_get_envvar`,
			Description: `Use this function when you need the value of a specific environment variable whose name you already know.

* NOTE:
- Make sure to report to the user if this function was called and the specified environment variable was successfully retrieved.`,
			InputSchema: &jsonschema.Schema{
				Type:     "object",
				ReadOnly: true,
				Properties: map[string]*jsonschema.Schema{
					"name": {
						Title:       "name",
						Description: `The name of an environment variable.`,
						Type:        "string",
					},
				},
				Required: []string{"name"},
			},
			Annotations: &mcp.ToolAnnotations{
				DestructiveHint: ptr(true),
				ReadOnlyHint:    true,
			},
		},
		handler: func(
			ctx context.Context,
			request *mcp.CallToolRequest,
		) (*mcp.CallToolResult, error) {
			var args map[string]any
			if err := json.Unmarshal(request.Params.Arguments, &args); err != nil {
				return mcpErrorResult("Failed to parse arguments: %s", err)
			}
			name, err := funcArg[string](args, "name")
			if err != nil || name == nil {
				return mcpErrorResult("Failed to get required argument 'name': %s", err)
			}
			marshalled, err := json.Marshal(map[string]string{
				"name":  *name,
				"value": os.Getenv(*name),
			})
			if err != nil {
				return mcpErrorResult("Failed to marshal environment variable: %s", err)
			}
			return mcpJSONResult(marshalled)
		},
	})

	// oll_stat_file (read only)
	toolsAndHandlers = append(toolsAndHandlers, toolAndHandler{
		tool: mcp.Tool{
			Name: `oll_stat_file`,
			Description: `Use this function when you need to check whether a file or directory exists or inspect its metadata, for example before accessing or handling it.

* NOTE:
- It is advised to call this function before accessing or handling files and/or directories.`,
			InputSchema: &jsonschema.Schema{
				Type:     "object",
				ReadOnly: true,
				Properties: map[string]*jsonschema.Schema{
					"filepath": {
						Title:       "filepath",
						Description: `An absolute path to a local file or directory.`,
						Type:        "string",
					},
				},
				Required: []string{"filepath"},
			},
			Annotations: &mcp.ToolAnnotations{
				ReadOnlyHint: true,
			},
		},
		handler: func(
			ctx context.Context,
			request *mcp.CallToolRequest,
		) (*mcp.CallToolResult, error) {
			var args map[string]any
			if err := json.Unmarshal(request.Params.Arguments, &args); err != nil {
				return mcpErrorResult("Failed to parse arguments: %s", err)
			}
			fpath, err := funcArg[string](args, "filepath")
			if err != nil || fpath == nil {
				return mcpErrorResult("Failed to get required argument 'filepath': %s", err)
			}
			stat, err := os.Stat(expandPath(*fpath))
			if err != nil {
				return mcpErrorResult("Failed to stat file: %s", err)
			}
			return mcpJSONResult([]byte(fileInfoToJSON(stat, expandPath(*fpath))))
		},
	})

	// oll_get_mimetype (read only)
	toolsAndHandlers = append(toolsAndHandlers, toolAndHandler{
		tool: mcp.Tool{
			Name: `oll_get_mimetype`,
			Description: `Use this function when you need to determine a file's mime type, for example before deciding how to read or process it.

* NOTE:
- It is advised to call this function before reading a file.`,
			InputSchema: &jsonschema.Schema{
				Type:     "object",
				ReadOnly: true,
				Properties: map[string]*jsonschema.Schema{
					"filepath": {
						Title:       "filepath",
						Description: `An absolute path to a local file.`,
						Type:        "string",
					},
				},
				Required: []string{"filepath"},
			},
			Annotations: &mcp.ToolAnnotations{
				ReadOnlyHint: true,
			},
		},
		handler: func(
			ctx context.Context,
			request *mcp.CallToolRequest,
		) (*mcp.CallToolResult, error) {
			var args map[string]any
			if err := json.Unmarshal(request.Params.Arguments, &args); err != nil {
				return mcpErrorResult("Failed to parse arguments: %s", err)
			}
			fpath, err := funcArg[string](args, "filepath")
			if err != nil || fpath == nil {
				return mcpErrorResult("Failed to get required argument 'filepath': %s", err)
			}
			mime, err := mimetype.DetectFile(expandPath(*fpath))
			if err != nil {
				return mcpErrorResult("Failed to get mime type: %s", err)
			}
			marshalled, err := json.Marshal(struct {
				Filepath  string `json:"filepath"`
				MimeType  string `json:"mimeType"`
				Extension string `json:"extension"`
			}{
				Filepath:  expandPath(*fpath),
				MimeType:  mime.String(),
				Extension: mime.Extension(),
			})
			if err != nil {
				return mcpErrorResult("Failed to marshal mime type: %s", err)
			}
			return mcpJSONResult(marshalled)
		},
	})

	// oll_list_files (read only)
	toolsAndHandlers = append(toolsAndHandlers, toolAndHandler{
		tool: mcp.Tool{
			Name:        `oll_list_files`,
			Description: `Use this function when you need to see the contents of a directory, for example before reading or selecting files within it.`,
			InputSchema: &jsonschema.Schema{
				Type:     "object",
				ReadOnly: true,
				Properties: map[string]*jsonschema.Schema{
					"dirpath": {
						Title:       "dirpath",
						Description: `An absolute path to a local directory.`,
						Type:        "string",
					},
				},
				Required: []string{"dirpath"},
			},
			Annotations: &mcp.ToolAnnotations{
				ReadOnlyHint: true,
			},
		},
		handler: func(
			ctx context.Context,
			request *mcp.CallToolRequest,
		) (*mcp.CallToolResult, error) {
			var args map[string]any
			if err := json.Unmarshal(request.Params.Arguments, &args); err != nil {
				return mcpErrorResult("Failed to parse arguments: %s", err)
			}
			dirpath, err := funcArg[string](args, "dirpath")
			if err != nil || dirpath == nil {
				return mcpErrorResult("Failed to get required argument 'dirpath': %s", err)
			}
			entries, err := os.ReadDir(expandPath(*dirpath))
			if err != nil {
				return mcpErrorResult("Failed to list files: %s", err)
			}
			return mcpJSONResult([]byte(dirEntriesToJSON(entries, expandPath(*dirpath))))
		},
	})

	// oll_read_text_file (read only, destructive)
	toolsAndHandlers = append(toolsAndHandlers, toolAndHandler{
		tool: mcp.Tool{
			Name: `oll_read_text_file`,
			Description: `Use this function when you need to read the contents of a plain text file at a given filepath.

* NOTE:
- Make sure to report to the user if this function was called and the specified file was successfully read.`,
			InputSchema: &jsonschema.Schema{
				Type:     "object",
				ReadOnly: true,
				Properties: map[string]*jsonschema.Schema{
					"filepath": {
						Title:       "filepath",
						Description: `An absolute path of a file that will be read.`,
						Type:        "string",
					},
				},
				Required: []string{"filepath"},
			},
			Annotations: &mcp.ToolAnnotations{
				DestructiveHint: ptr(true),
				ReadOnlyHint:    true,
			},
		},
		handler: func(
			ctx context.Context,
			request *mcp.CallToolRequest,
		) (*mcp.CallToolResult, error) {
			var args map[string]any
			if err := json.Unmarshal(request.Params.Arguments, &args); err != nil {
				return mcpErrorResult("Failed to parse arguments: %s", err)
			}
			fpath, err := funcArg[string](args, "filepath")
			if err != nil || fpath == nil {
				return mcpErrorResult("Failed to get required argument 'filepath': %s", err)
			}
			content, err := os.ReadFile(expandPath(*fpath))
			if err != nil {
				return mcpErrorResult("Failed to read file: %s", err)
			}
			if mime := mimetype.Detect(content); !mime.Is("text/plain") {
				return mcpErrorResult("given file '%s' is not in text/plain format: %s", expandPath(*fpath), mime.String())
			}
			marshalled, err := json.Marshal(struct {
				Filepath string `json:"filepath"`
				Content  string `json:"content"`
			}{
				Filepath: expandPath(*fpath),
				Content:  string(content),
			})
			if err != nil {
				return mcpErrorResult("Failed to marshal read file: %s", err)
			}
			return mcpJSONResult(marshalled)
		},
	})

	// oll_create_text_file (destructive)
	toolsAndHandlers = append(toolsAndHandlers, toolAndHandler{
		tool: mcp.Tool{
			Name: `oll_create_text_file`,
			Description: `Use this function when you need to write a new plain text file at a given filepath.

* CAUTION:
- There should not be an existing file at the given path.
- This function should not be used for creating binary files due to the risk of file corruption.

* NOTE:
- Make sure to report to the user if this function was called and the specified file was successfully created.`,
			InputSchema: &jsonschema.Schema{
				Type:     "object",
				ReadOnly: true,
				Properties: map[string]*jsonschema.Schema{
					"content": {
						Title:       "content",
						Description: "A plain text content of a file that will be newly created.",
						Type:        "string",
					},
					"filepath": {
						Title:       "filepath",
						Description: `An absolute path of a file that will be newly created.`,
						Type:        "string",
					},
				},
				Required: []string{"content", "filepath"},
			},
			Annotations: &mcp.ToolAnnotations{
				DestructiveHint: ptr(true),
			},
		},
		handler: func(
			ctx context.Context,
			request *mcp.CallToolRequest,
		) (*mcp.CallToolResult, error) {
			var args map[string]any
			if err := json.Unmarshal(request.Params.Arguments, &args); err != nil {
				return mcpErrorResult("Failed to parse arguments: %s", err)
			}
			fpath, err := funcArg[string](args, "filepath")
			if err != nil || fpath == nil {
				return mcpErrorResult("Failed to get required argument 'filepath': %s", err)
			}
			content, err := funcArg[string](args, "content")
			if err != nil || content == nil {
				return mcpErrorResult("Failed to get required argument 'content': %s", err)
			}
			if err := os.WriteFile(expandPath(*fpath), []byte(*content), 0o644); err != nil {
				return mcpErrorResult("Failed to create text file: %s", err)
			}
			return mcpTextResult(fmt.Sprintf("File was successfully created at path: '%s'", expandPath(*fpath)))
		},
	})

	// oll_delete_file (destructive)
	toolsAndHandlers = append(toolsAndHandlers, toolAndHandler{
		tool: mcp.Tool{
			Name: `oll_delete_file`,
			Description: `Use this function when you need to remove a file at a given filepath.

* NOTE:
- Make sure to report to the user if this function was called and the specified file was successfully deleted.`,
			InputSchema: &jsonschema.Schema{
				Type:     "object",
				ReadOnly: true,
				Properties: map[string]*jsonschema.Schema{
					"filepath": {
						Title:       "filepath",
						Description: `An absolute path of a file that will be deleted.`,
						Type:        "string",
					},
				},
				Required: []string{"filepath"},
			},
			Annotations: &mcp.ToolAnnotations{
				DestructiveHint: ptr(true),
			},
		},
		handler: func(
			ctx context.Context,
			request *mcp.CallToolRequest,
		) (*mcp.CallToolResult, error) {
			var args map[string]any
			if err := json.Unmarshal(request.Params.Arguments, &args); err != nil {
				return mcpErrorResult("Failed to parse arguments: %s", err)
			}
			fpath, err := funcArg[string](args, "filepath")
			if err != nil || fpath == nil {
				return mcpErrorResult("Failed to get required argument 'filepath': %s", err)
			}
			if err := os.Remove(expandPath(*fpath)); err != nil {
				return mcpErrorResult("Failed to delete file: %s", err)
			}
			return mcpTextResult(fmt.Sprintf("File was successfully deleted: '%s'", expandPath(*fpath)))
		},
	})

	// oll_move_file (destructive)
	toolsAndHandlers = append(toolsAndHandlers, toolAndHandler{
		tool: mcp.Tool{
			Name: `oll_move_file`,
			Description: `Use this function when you need to move or rename a file from one filepath to another.

* NOTE:
- Make sure to report to the user if this function was called and the specified file was successfully moved.`,
			InputSchema: &jsonschema.Schema{
				Type:     "object",
				ReadOnly: true,
				Properties: map[string]*jsonschema.Schema{
					"from": {
						Title:       "from",
						Description: `An original path (absolute) of a file that will be moved.`,
						Type:        "string",
					},
					"to": {
						Title:       "to",
						Description: `A destination path (absolute) of a moved file.`,
						Type:        "string",
					},
				},
				Required: []string{"from", "to"},
			},
			Annotations: &mcp.ToolAnnotations{
				DestructiveHint: ptr(true),
			},
		},
		handler: func(
			ctx context.Context,
			request *mcp.CallToolRequest,
		) (*mcp.CallToolResult, error) {
			var args map[string]any
			if err := json.Unmarshal(request.Params.Arguments, &args); err != nil {
				return mcpErrorResult("Failed to parse arguments: %s", err)
			}
			from, err := funcArg[string](args, "from")
			if err != nil || from == nil {
				return mcpErrorResult("Failed to get required argument 'from': %s", err)
			}
			to, err := funcArg[string](args, "to")
			if err != nil || to == nil {
				return mcpErrorResult("Failed to get required argument 'to': %s", err)
			}
			if err := os.Rename(expandPath(*from), expandPath(*to)); err != nil {
				return mcpErrorResult("Failed to move file: %s", err)
			}
			return mcpTextResult(fmt.Sprintf("File was successfully moved: '%s' -> '%s'", expandPath(*from), expandPath(*to)))
		},
	})

	// oll_run_cmdline (destructive)
	toolsAndHandlers = append(toolsAndHandlers, toolAndHandler{
		tool: mcp.Tool{
			Name: `oll_run_cmdline`,
			Description: fmt.Sprintf(`Use this function when there is no other tool available to accomplish the task and you need to execute a bash commandline to get its output.

* RULES:
- The commandline must be in one line, and should be escaped correctly.
- This function should be treated as a last resort when there is no other way for you to fulfill the given task.

* CAUTION:
- Never pass malicious input or non-existing commands to this function, as it will be executed as a shell command.
- This function will fail with timeout if the commandline takes %d seconds or longer to finish.`, commandTimeoutSeconds),
			InputSchema: &jsonschema.Schema{
				Type:     "object",
				ReadOnly: true,
				Properties: map[string]*jsonschema.Schema{
					"cmdline": {
						Title:       "cmdline",
						Description: `A bash commandline.`,
						Type:        "string",
					},
				},
				Required: []string{"cmdline"},
			},
			Annotations: &mcp.ToolAnnotations{
				DestructiveHint: ptr(true),
			},
		},
		handler: func(
			ctx context.Context,
			request *mcp.CallToolRequest,
		) (*mcp.CallToolResult, error) {
			var args map[string]any
			if err := json.Unmarshal(request.Params.Arguments, &args); err != nil {
				return mcpErrorResult("Failed to parse arguments: %s", err)
			}
			cmdline, err := funcArg[string](args, "cmdline")
			if err != nil || cmdline == nil {
				return mcpErrorResult("Failed to get required argument 'cmdline': %s", err)
			}
			command, cmdArgs, err := parseCommandline(*cmdline)
			if err != nil {
				return mcpErrorResult("Failed to parse 'cmdline': %s", err)
			}
			cmdCtx, cancel := context.WithTimeout(ctx, commandTimeoutSeconds*time.Second)
			defer cancel()
			stdout, stderr, exit, runErr := runCommandWithContext(cmdCtx, command, cmdArgs...)
			marshalled, mErr := json.Marshal(struct {
				Cmdline  string `json:"cmdline"`
				ExitCode int    `json:"exitCode"`
				Output   string `json:"output,omitempty"`
				Error    string `json:"error,omitempty"`
			}{
				Cmdline:  *cmdline,
				ExitCode: exit,
				Output:   stdout,
				Error:    stderr,
			})
			if mErr != nil {
				return mcpErrorResult("Failed to marshal cmdline result: %s", mErr)
			}
			_ = runErr // non-zero exit is reported via exitCode/stderr, not as a tool error
			return mcpJSONResult(marshalled)
		},
	})

	// oll_do_http (destructive)
	toolsAndHandlers = append(toolsAndHandlers, toolAndHandler{
		tool: mcp.Tool{
			Name:        `oll_do_http`,
			Description: `Use this function when you need to send an HTTP request to a URL and read its response, for example to call a web API or fetch remote content.`,
			Annotations: &mcp.ToolAnnotations{
				DestructiveHint: ptr(true),
			},
			InputSchema: &jsonschema.Schema{
				Type:     "object",
				ReadOnly: true,
				Properties: map[string]*jsonschema.Schema{
					"method": {
						Title:       "method",
						Description: `HTTP request method.`,
						Type:        "string",
						Enum:        []any{"GET", "DELETE", "HEAD", "POST", "PUT", "PATCH"},
					},
					"url": {
						Title:       "url",
						Description: `HTTP request URL.`,
						Type:        "string",
					},
					"headers": {
						Title:       "headers",
						Description: `HTTP request headers. Keys and values are all strings.`,
						Type:        "object",
					},
					"params": {
						Title:       "params",
						Description: `HTTP request parameters, especially for GET/DELETE requests.`,
						Type:        "object",
					},
					"body": {
						Title: "body",
						Description: `HTTP request body, especially for POST/PUT requests.

* NOTE:
Mime type of this parameter should also be specified in the 'Content-Type' header, eg. 'application/json', with the 'headers' parameter.`,
						Type: "string",
					},
				},
				Required: []string{"method", "url"},
			},
		},
		handler: func(
			ctx context.Context,
			request *mcp.CallToolRequest,
		) (*mcp.CallToolResult, error) {
			var args map[string]any
			if err := json.Unmarshal(request.Params.Arguments, &args); err != nil {
				return mcpErrorResult("Failed to parse arguments: %s", err)
			}
			method, err := funcArg[string](args, "method")
			if err != nil || method == nil {
				return mcpErrorResult("Failed to get required argument 'method': %s", err)
			}
			urlString, err := funcArg[string](args, "url")
			if err != nil || urlString == nil {
				return mcpErrorResult("Failed to get required argument 'url': %s", err)
			}
			u, err := url.Parse(*urlString)
			if err != nil {
				return mcpErrorResult("Invalid url '%s': %s", *urlString, err)
			}
			headers, _ := funcArg[map[string]any](args, "headers")
			reqParams, _ := funcArg[map[string]any](args, "params")
			body, _ := funcArg[string](args, "body")

			var req *http.Request
			switch *method {
			case "GET", "DELETE", "HEAD":
				req, err = http.NewRequestWithContext(ctx, *method, u.String(), nil)
				if err == nil && reqParams != nil {
					q := req.URL.Query()
					for k, v := range *reqParams {
						q.Add(k, fmt.Sprintf("%v", v))
					}
					req.URL.RawQuery = q.Encode()
				}
			case "POST", "PUT", "PATCH":
				var reader *bytes.Reader
				if body != nil {
					reader = bytes.NewReader([]byte(*body))
				} else {
					reader = bytes.NewReader(nil)
				}
				req, err = http.NewRequestWithContext(ctx, *method, u.String(), reader)
			default:
				return mcpErrorResult("Not a supported 'method' for http request: '%s'", *method)
			}
			if err != nil {
				return mcpErrorResult("Failed to build http request: %s", err)
			}
			if headers != nil {
				for k, v := range *headers {
					req.Header.Set(k, fmt.Sprintf("%v", v))
				}
			}

			resp, err := http.DefaultClient.Do(req)
			if err != nil {
				return mcpErrorResult("Failed to do http request: %s", err)
			}
			defer func() { _ = resp.Body.Close() }()
			respBody, err := io.ReadAll(resp.Body)
			if err != nil {
				return mcpErrorResult("Failed to read http response: %s", err)
			}
			marshalled, err := json.Marshal(struct {
				RequestedMethod string              `json:"requestedMethod"`
				RequestedURL    string              `json:"requestedUrl"`
				Status          int                 `json:"status"`
				Headers         map[string][]string `json:"headers,omitempty"`
				Body            string              `json:"body"`
			}{
				RequestedMethod: *method,
				RequestedURL:    *urlString,
				Status:          resp.StatusCode,
				Headers:         resp.Header,
				Body:            string(respBody),
			})
			if err != nil {
				return mcpErrorResult("Failed to marshal http response: %s", err)
			}
			return mcpJSONResult(marshalled)
		},
	})

	// add tools to server
	tools := []*mcp.Tool{}
	for _, t := range toolsAndHandlers {
		server.AddTool(&t.tool, t.handler)
		tools = append(tools, &t.tool)
	}

	return server, tools
}

// selfGenerateHandler returns the handler for the oll_generate tool.
func selfGenerateHandler(
	output *outputWriter,
	conf config,
	p params,
) mcp.ToolHandler {
	return func(
		ctx context.Context,
		request *mcp.CallToolRequest,
	) (*mcp.CallToolResult, error) {
		var args map[string]any
		if err := json.Unmarshal(request.Params.Arguments, &args); err != nil {
			return mcpErrorResult("Failed to parse arguments: %s", err)
		}

		prompt, err := funcArg[string](args, "prompt")
		if err != nil || prompt == nil {
			return mcpErrorResult("Failed to get required argument 'prompt': %s", err)
		}
		modality, err := funcArg[string](args, "modality")
		if err != nil || modality == nil {
			return mcpErrorResult("Failed to get required argument 'modality': %s", err)
		}

		// filepaths (JSON array -> []any of strings)
		var filepaths []*string
		if fps, _ := funcArg[[]any](args, "filepaths"); fps != nil {
			for _, fp := range *fps {
				if s, ok := fp.(string); ok {
					filepaths = append(filepaths, ptr(expandPath(s)))
				}
			}
		}

		// model (with modality-specific defaults)
		model, _ := funcArg[string](args, "model")

		ctx, cancel := context.WithTimeout(ctx, mcpFunctionTimeoutSeconds*time.Second)
		defer cancel()

		switch *modality {
		case "text":
			if model == nil {
				if conf.DefaultModel != nil {
					model = conf.DefaultModel
				} else {
					model = ptr(string(defaultModel))
				}
			}
			return selfGenerateText(ctx, output, conf, p, *model, *prompt, filepaths, args)
		case "image":
			if model == nil {
				if conf.ImageGenerationModel != nil {
					model = conf.ImageGenerationModel
				} else {
					model = ptr(string(defaultModelForImageGeneration))
				}
			}
			return selfGenerateImage(ctx, output, conf, p, *model, *prompt, filepaths, args)
		default:
			return mcpErrorResult("Unsupported modality: '%s' (must be 'text' or 'image')", *modality)
		}
	}
}

// selfGenerateText generates text and returns the accumulated content as an MCP result.
func selfGenerateText(
	ctx context.Context,
	output *outputWriter,
	conf config,
	p params,
	model, prompt string,
	filepaths []*string,
	args map[string]any,
) (*mcp.CallToolResult, error) {
	client, err := newOllamaClient()
	if err != nil {
		return mcpErrorResult("Failed to initialize Ollama API client: %s", err)
	}

	// model capability (thinking)
	shown, err := client.Show(ctx, &api.ShowRequest{Model: model})
	if err != nil {
		return mcpErrorResult("Failed to get model(%s) info: %s", model, err)
	}
	withThinking := false
	if wt, _ := funcArg[bool](args, "with_thinking"); wt != nil {
		withThinking = *wt
	}
	var thinkVal any = false
	if slices.Contains(shown.Capabilities, mdl.CapabilityThinking) {
		thinkVal = withThinking
	}

	// convert_url
	convertURL := false
	if cu, _ := funcArg[bool](args, "convert_url"); cu != nil {
		convertURL = *cu
	}
	filesInPrompt := map[string][]byte{}
	if convertURL {
		userAgent := ptr(defaultUserAgent)
		if p.UserAgent != nil {
			userAgent = p.UserAgent
		}
		prompt, filesInPrompt = replaceURLsInPrompt(output, conf, userAgent, prompt, p.Verbose)
	}

	// system instruction
	systemInstruction := defaultSystemInstruction(paramsWithModel(p, model))
	if p.Generation.DetailedOptions.SystemInstruction != nil {
		systemInstruction = *p.Generation.DetailedOptions.SystemInstruction
	} else if conf.SystemInstruction != nil {
		systemInstruction = *conf.SystemInstruction
	}

	// convert prompt + files
	prompt, mediaFiles, err := convertPromptAndFiles(prompt, filesInPrompt, filepaths)
	if err != nil {
		return mcpErrorResult("Failed to convert prompt and files: %s", err)
	}
	var media []api.ImageData
	for _, mf := range mediaFiles {
		media = append(media, api.ImageData(mf))
	}

	req := &api.ChatRequest{
		Model: model,
		Messages: []api.Message{
			{Role: "system", Content: systemInstruction},
			{Role: "user", Content: prompt, Images: media},
		},
		Options: map[string]any{
			"temperature": defaultGenerationTemperature,
			"top_p":       defaultGenerationTopP,
			"top_k":       defaultGenerationTopK,
		},
		Think: &api.ThinkValue{Value: thinkVal},
	}

	var sb strings.Builder
	if err := client.Chat(ctx, req, func(resp api.ChatResponse) error {
		if resp.Message.Role == "assistant" && len(resp.Message.Content) > 0 {
			sb.WriteString(resp.Message.Content)
		}
		return nil
	}); err != nil {
		return mcpErrorResult("Failed to generate: %s", err)
	}

	return mcpTextResult(sb.String())
}

// selfGenerateImage generates an image, saves it to a file, and returns the filepath.
func selfGenerateImage(
	ctx context.Context,
	output *outputWriter,
	conf config,
	p params,
	model, prompt string,
	filepaths []*string,
	args map[string]any,
) (*mcp.CallToolResult, error) {
	client, err := newOllamaClient()
	if err != nil {
		return mcpErrorResult("Failed to initialize Ollama API client: %s", err)
	}

	shown, err := client.Show(ctx, &api.ShowRequest{Model: model})
	if err != nil {
		return mcpErrorResult("Failed to get model(%s) info: %s", model, err)
	}
	if !slices.Contains(shown.Capabilities, mdl.CapabilityImage) {
		return mcpErrorResult("model(%s) does not support image generation", model)
	}

	// dimensions + seed
	width := defaultImageGenerationWidth
	if w, _ := funcArg[float64](args, "image_width"); w != nil {
		width = int(*w)
	}
	height := defaultImageGenerationHeight
	if h, _ := funcArg[float64](args, "image_height"); h != nil {
		height = int(*h)
	}
	seed := rand.Int()
	if s, _ := funcArg[float64](args, "image_seed"); s != nil {
		seed = int(*s)
	}

	// convert prompt + files
	prompt, mediaFiles, err := convertPromptAndFiles(prompt, nil, filepaths)
	if err != nil {
		return mcpErrorResult("Failed to convert prompt and files: %s", err)
	}
	var media []api.ImageData
	if len(mediaFiles) > 0 {
		media = append(media, mediaFiles...)
	}

	req := &api.GenerateRequest{
		Model:  model,
		Prompt: prompt,
		Images: media,
		Width:  int32(width),
		Height: int32(height),
		Steps:  int32(defaultImageGenerationSteps),
		Options: map[string]any{
			"seed": seed,
		},
		KeepAlive: &api.Duration{
			Duration: time.Duration(conf.ImageGenerationTimeoutSeconds) * time.Second,
		},
	}

	var imageBase64 string
	if err := client.Generate(ctx, req, func(resp api.GenerateResponse) error {
		if resp.Done {
			if resp.Image != "" {
				imageBase64 = resp.Image
			} else if strings.HasPrefix(resp.Response, "IMAGE_BASE64:") {
				imageBase64 = resp.Response[13:]
			}
		}
		return nil
	}); err != nil {
		return mcpErrorResult("Failed to generate image: %s", err)
	}

	if imageBase64 == "" {
		return mcpErrorResult("Failed to generate image: no image data received")
	}

	imageData, err := base64.StdEncoding.DecodeString(imageBase64)
	if err != nil {
		return mcpErrorResult("Failed to decode image: %s", err)
	}

	timestamp := time.Now().Format("20060102_150405")
	filename := fmt.Sprintf("oll_%s.png", timestamp)
	fpath := filepath.Join(imagesSaveDir(p.Generation.Image.SaveImagesToDir), filename)
	if err := os.WriteFile(fpath, imageData, 0o644); err != nil {
		return mcpErrorResult("Failed to save image: %s", err)
	}

	return mcpTextResult(fmt.Sprintf(
		"Generated an image (%d bytes, %s, seed %d) and saved it to: '%s'",
		len(imageData), mimetype.Detect(imageData).String(), seed, fpath,
	))
}

// paramsWithModel returns a copy of p with Model set, for defaultSystemInstruction.
func paramsWithModel(p params, model string) params {
	p.Model = ptr(model)
	return p
}

// selfAsMCPTool runs oll itself as an in-memory MCP tool and returns its connection + tools.
func selfAsMCPTool(
	ctx context.Context,
	conf config,
	p params,
	output *outputWriter,
) (connection *mcp.ClientSession, tools []*mcp.Tool, err error) {
	server, tools := buildSelfServer(output, conf, p)

	output.verbose(
		verboseMinimum,
		p.Verbose,
		"connecting to local MCP server (self)...",
	)

	if connection, err = mcpRunInMemory(ctx, server); err != nil {
		return nil, nil, fmt.Errorf("failed to run in-memory MCP server (self): %w", err)
	}
	return connection, tools, nil
}

// runStdioServer serves oll as a STDIO MCP server until interrupted.
func runStdioServer(
	ctx context.Context,
	output *outputWriter,
	conf config,
	p params,
) (err error) {
	server, _ := buildSelfServer(output, conf, p)

	ctx, cancel := signal.NotifyContext(ctx, os.Interrupt, syscall.SIGTERM)
	defer cancel()

	output.verbose(verboseMinimum, p.Verbose, "starting STDIO MCP server...")

	if _, err = server.Connect(
		ctx,
		&mcp.StdioTransport{},
		&mcp.ServerSessionOptions{},
	); err != nil {
		if err == context.Canceled {
			return nil
		}
		return fmt.Errorf("server connection error: %w", err)
	}

	<-ctx.Done()
	return nil
}

// serve reads config and runs oll as a standalone STDIO MCP server.
func serve(
	output *outputWriter,
	p params,
) (exit int, err error) {
	var conf config
	if conf, err = readConfig(resolveConfigFilepath(p.ConfigFilepath)); err != nil {
		return 1, fmt.Errorf("failed to read configuration: %w", err)
	}

	if err = runStdioServer(context.TODO(), output, conf, p); err != nil {
		return 1, err
	}
	return 0, nil
}

type fileInfo struct {
	Name         string      `json:"filename"`
	AbsolutePath string      `json:"absolutePath"`
	Size         int64       `json:"filesize"`
	Mode         os.FileMode `json:"mode"`
	Modified     time.Time   `json:"modified"`
	IsDirectory  bool        `json:"isDirectory"`
	Sys          string      `json:"sys"`
}

// fileInfoToStruct converts os.FileInfo to a fileInfo struct.
func fileInfoToStruct(info os.FileInfo, absoluteFilepath string) fileInfo {
	return fileInfo{
		Name:         info.Name(),
		AbsolutePath: absoluteFilepath,
		Size:         info.Size(),
		Mode:         info.Mode(),
		Modified:     info.ModTime(),
		IsDirectory:  info.IsDir(),
		Sys:          fmt.Sprintf("%#v", info.Sys()),
	}
}

// fileInfoToJSON converts os.FileInfo to a JSON string.
func fileInfoToJSON(info os.FileInfo, absoluteFilepath string) string {
	result := fileInfoToStruct(info, absoluteFilepath)
	if marshalled, err := json.Marshal(result); err == nil {
		return string(marshalled)
	}
	return fmt.Sprintf("%+v", result)
}

type dirEntry struct {
	Name        string      `json:"filename"`
	IsDirectory bool        `json:"isDirectory"`
	Mode        os.FileMode `json:"mode"`

	Info *fileInfo `json:"info,omitempty"`
}

// dirEntryToStruct converts os.DirEntry to a dirEntry struct.
func dirEntryToStruct(entry os.DirEntry, parentDirpath string) dirEntry {
	result := dirEntry{
		Name:        entry.Name(),
		IsDirectory: entry.IsDir(),
		Mode:        entry.Type(),
	}
	if info, _ := entry.Info(); info != nil { // may be nil when entry was removed after listing
		result.Info = ptr(fileInfoToStruct(info, filepath.Join(parentDirpath, entry.Name())))
	}
	return result
}

// dirEntriesToJSON converts os.DirEntry slice to a JSON string.
func dirEntriesToJSON(entries []os.DirEntry, parentDirpath string) string {
	result := []dirEntry{}
	for _, entry := range entries {
		result = append(result, dirEntryToStruct(entry, parentDirpath))
	}
	if marshalled, err := json.Marshal(struct {
		Result []dirEntry `json:"result"`
	}{Result: result}); err == nil {
		return string(marshalled)
	}
	return fmt.Sprintf("%+v", result)
}

// runCommandWithContext runs the given command + args with context.
func runCommandWithContext(
	ctx context.Context,
	command string,
	args ...string,
) (stdout, stderr string, exitCode int, err error) {
	cmd := exec.CommandContext(ctx, command, args...)

	var stdoutBuf, stderrBuf bytes.Buffer
	cmd.Stdout = &stdoutBuf
	cmd.Stderr = &stderrBuf

	err = cmd.Run()

	stdout = stdoutBuf.String()
	stderr = stderrBuf.String()
	exitCode = 0

	if err != nil {
		if exitErr, ok := err.(*exec.ExitError); ok {
			if status, ok := exitErr.Sys().(syscall.WaitStatus); ok {
				exitCode = status.ExitStatus()
			} else {
				exitCode = 1
				stderr += fmt.Sprintf("\n(Failed to get specific exit status, error: %v)\n", err)
			}
		} else {
			exitCode = 1
			stderr += fmt.Sprintf("\n(Command failed with non-ExitError: %v)\n", err)
		}
	}

	return stdout, stderr, exitCode, err
}
