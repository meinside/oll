// mcp.go

package main

import (
	"context"
	"encoding/json"
	"fmt"
	"net"
	"net/http"
	"os/exec"
	"strings"
	"time"

	"github.com/google/jsonschema-go/jsonschema"
	"github.com/meinside/version-go"
	"github.com/modelcontextprotocol/go-sdk/mcp"
	"github.com/ollama/ollama/api"
)

const (
	mcpClientName = `oll/mcp`

	mcpMaxRetries = 3
)

type mcpServerType string

const (
	mcpServerStreamable mcpServerType = "streamable"
	mcpServerStdio      mcpServerType = "stdio"
)

// a map of MCP connections and tools
type mcpConnectionsAndTools map[string]struct {
	serverType mcpServerType
	connection *mcp.ClientSession
	tools      []*mcp.Tool
}

// mcpToolFrom gets a matched server url and tool from given MCP tools and function name.
func mcpToolFrom(
	mcpConnsAndTools mcpConnectionsAndTools,
	fnName string,
) (serverKey string, serverType mcpServerType, mc *mcp.ClientSession, tool mcp.Tool, exists bool) {
	for serverKey, connsAndTools := range mcpConnsAndTools {
		for _, tool := range connsAndTools.tools {
			if tool != nil && tool.Name == fnName {
				return serverKey, connsAndTools.serverType, connsAndTools.connection, *tool, true
			}
		}
	}

	return "", "", nil, mcp.Tool{}, false
}

// keysFromTools extracts keys from given tools.
func keysFromTools(
	localTools []api.Tool,
	mcpConnsAndTools mcpConnectionsAndTools,
) (localToolKeys, mcpToolKeys []string) {
	for _, tool := range localTools {
		localToolKeys = append(localToolKeys, tool.Function.Name)
	}
	for _, connsAndTools := range mcpConnsAndTools {
		for _, tool := range connsAndTools.tools {
			mcpToolKeys = append(mcpToolKeys, tool.Name)
		}
	}

	return localToolKeys, mcpToolKeys
}

// fetchMCPTools fetches function declarations from the MCP server.
func fetchMCPTools(
	ctx context.Context,
	mc *mcp.ClientSession,
) (tools []*mcp.Tool, err error) {
	var listed *mcp.ListToolsResult
	if listed, err = mc.ListTools(ctx, &mcp.ListToolsParams{}); err == nil {
		return listed.Tools, nil
	}
	return tools, err
}

// mcpToOllamaTools converts given MCP tools `from` to Ollama tools.
//
// InputSchema value of each mcp.Tool should be in type: `jsonschema.Schema` or `map[string]any`.
func mcpToOllamaTools(
	from []*mcp.Tool,
) (to []*api.Tool, err error) {
	to = make([]*api.Tool, len(from))

	for i, f := range from {
		to[i] = &api.Tool{
			Type: "function",
			Function: api.ToolFunction{
				Name:        f.Name,
				Description: f.Description,
			},
		}
		if inputSchema, ok := f.InputSchema.(jsonschema.Schema); ok {
			if marshalled, err := inputSchema.MarshalJSON(); err == nil {
				var schema map[string]any
				if err := json.Unmarshal(marshalled, &schema); err == nil {
					to[i].Items = schema
				} else {
					return nil, fmt.Errorf("could not convert json to map: %w", err)
				}
			} else {
				return nil, fmt.Errorf("could not convert input schema to json: %w", err)
			}
		} else if inputSchema, ok := f.InputSchema.(map[string]any); ok {
			to[i].Items = inputSchema
		} else {
			return nil, fmt.Errorf("tools[%d].InputSchema is not in type `jsonschema.Schema` or `map[string]any`: %T", i, f.InputSchema)
		}
	}

	return to, nil
}

// fetchToolCallResult fetches function result from MCP.
func fetchToolCallResult(
	ctx context.Context,
	mc *mcp.ClientSession,
	fnName string,
	fnArgs api.ToolCallFunctionArguments,
) (res *mcp.CallToolResult, err error) {
	if res, err = mc.CallTool(
		ctx,
		&mcp.CallToolParams{
			Name:      fnName,
			Arguments: fnArgs.ToMap(),
		},
	); err == nil {
		return res, nil
	}

	return res, err
}

// mcpConnect connects to MCP server, starts, initializes, and returns the client.
func mcpConnect(
	ctx context.Context,
	url string,
) (connection *mcp.ClientSession, err error) {
	streamable := &mcp.StreamableClientTransport{
		Endpoint:   url,
		HTTPClient: mcpHTTPClient(),
		MaxRetries: mcpMaxRetries,
	}

	client := mcp.NewClient(
		&mcp.Implementation{
			Name:    mcpClientName,
			Version: version.Build(version.OS | version.Architecture),
		},
		&mcp.ClientOptions{},
	)

	if connection, err = client.Connect(
		ctx,
		streamable,
		&mcp.ClientSessionOptions{},
	); err == nil {
		return connection, nil
	}

	return nil, err
}

// mcpRun launches MCP server with given `cmdline` and connects to it,
// then starts, initializes, and returns the client.
func mcpRun(
	ctx context.Context,
	cmdline string,
) (connection *mcp.ClientSession, err error) {
	cmdline = expandPath(cmdline)

	command, args, err := parseCommandline(cmdline)
	if err != nil {
		return nil, fmt.Errorf(
			"failed to parse command line `%s` %w",
			stripServerInfo(mcpServerStdio, cmdline),
			err,
		)
	}

	if connection, err = mcp.NewClient(
		&mcp.Implementation{
			Name:    mcpClientName,
			Version: version.Build(version.OS | version.Architecture),
		},
		&mcp.ClientOptions{},
	).Connect(
		ctx,
		&mcp.CommandTransport{
			Command: exec.Command(command, args...),
		},
		&mcp.ClientSessionOptions{},
	); err == nil {
		return connection, nil
	}

	return nil, err
}

const (
	mcpDefaultDialTimeoutSeconds           = 10
	mcpDefaultKeepAliveSeconds             = 60
	mcpDefaultIdleTimeoutSeconds           = 180
	mcpDefaultTLSHandshakeTimeoutSeconds   = 20
	mcpDefaultResponseHeaderTimeoutSeconds = 60
	mcpDefaultExpectContinueTimeoutSeconds = 15
)

// for reusing http client
var _mcpHTTPClient *http.Client

// mcpHTTPClient generates a http client for MCP connection.
func mcpHTTPClient() *http.Client {
	if _mcpHTTPClient == nil {
		_mcpHTTPClient = &http.Client{
			Timeout: defaultTimeoutSeconds * time.Second,
			Transport: &http.Transport{
				DialContext: (&net.Dialer{
					Timeout:   mcpDefaultDialTimeoutSeconds * time.Second,
					KeepAlive: mcpDefaultKeepAliveSeconds * time.Second,
				}).DialContext,
				IdleConnTimeout:       mcpDefaultIdleTimeoutSeconds * time.Second,
				TLSHandshakeTimeout:   mcpDefaultTLSHandshakeTimeoutSeconds * time.Second,
				ResponseHeaderTimeout: mcpDefaultResponseHeaderTimeoutSeconds * time.Second,
				ExpectContinueTimeout: mcpDefaultExpectContinueTimeoutSeconds * time.Second,
				DisableCompression:    true,
			},
		}
	}
	return _mcpHTTPClient
}

// stripServerInfo strips sensitive information from given server info.
func stripServerInfo(serverType mcpServerType, info string) string {
	switch serverType {
	case mcpServerStreamable:
		return strings.Split(info, "?")[0]
	case mcpServerStdio:
		cmd, _, _ := parseCommandline(info)
		return cmd
	}
	return info
}
