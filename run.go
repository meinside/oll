// run.go

package main

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"time"

	"github.com/fatih/color"
	"github.com/jessevdk/go-flags"
	"github.com/meinside/version-go"
	"github.com/modelcontextprotocol/go-sdk/mcp"
	"github.com/ollama/ollama/api"
)

const (
	defaultConfigFilename = "config.json"

	defaultSystemInstructionFormat = `You are a CLI named '%[1]s' which uses (s)LLM model '%[2]s'.

Current datetime is %[3]s, and hostname is '%[4]s'.

Respond to user messages according to the following principles:
- Do not repeat the user's request and return only the response to the user's request.
- Unless otherwise specified, respond in the same language as used in the user's request.
- Be as accurate as possible.
- Be as truthful as possible.
- Be as comprehensive and informative as possible.
- When textual files are provided for context, they will be listed between the '` + filesTagBegin + filesTagEnd + `' tags in the prompt, so make sure to use them if provided.
`

	defaultTimeoutSeconds         = 5 * 60 // 5 minutes
	defaultFetchURLTimeoutSeconds = 10     // 10 seconds
	defaultUserAgent              = `oll/fetcher`
)

// run the application with params
func run(
	output *outputWriter,
	parser *flags.Parser,
	p params,
) (exitCode int, err error) {
	// early return if no task was requested
	if !p.taskRequested() {
		output.error("No task was requested")

		return output.printHelpBeforeExit(1, parser), nil
	}

	// early return after printing the version
	if p.ShowVersion {
		output.printColored(
			color.FgGreen,
			"%s %s\n\n",
			appName,
			version.Build(version.OS|version.Architecture),
		)

		return output.printHelpBeforeExit(0, parser), nil
	}

	// read and apply configs
	var conf config
	if conf, err = readConfig(resolveConfigFilepath(p.ConfigFilepath)); err == nil {
		if p.Generation.SystemInstruction == nil && conf.SystemInstruction != nil {
			p.Generation.SystemInstruction = conf.SystemInstruction
		}
	} else {
		return 1, fmt.Errorf("failed to read configuration: %w", err)
	}

	// override parameters with command arguments
	if conf.DefaultModel != nil && p.Model == nil {
		p.Model = conf.DefaultModel
	}

	// set default values
	if p.Model == nil {
		p.Model = ptr(defaultModel)
	}
	if p.Generation.SystemInstruction == nil {
		p.Generation.SystemInstruction = ptr(defaultSystemInstruction(p))
	}
	if p.UserAgent == nil {
		p.UserAgent = ptr(defaultUserAgent)
	}

	// expand filepaths (recurse directories)
	p.Generation.Filepaths, err = expandFilepaths(output, p)
	if err != nil {
		return 1, fmt.Errorf("failed to read given filepaths: %w", err)
	}

	if p.hasPrompt() { // if prompt is given,
		if p.Embeddings.GenerateEmbeddings {
			output.verbose(
				verboseMaximum,
				p.Verbose,
				"embeddings request params with prompt: %s\n\n",
				prettify(p),
			)

			return doEmbeddingsGeneration(context.TODO(), output, conf, p)
		} else {
			output.verbose(
				verboseMaximum,
				p.Verbose,
				"generation request params with prompt: %s\n\n",
				prettify(p),
			)

			// tools (local) and callbacks
			var localTools []api.Tool = nil
			if p.LocalTools.Tools != nil {
				if bytes, err := standardizeJSON([]byte(*p.LocalTools.Tools)); err != nil {
					return 1, fmt.Errorf("failed to standardize json for local tool: %w", err)
				} else {
					if err := json.Unmarshal(bytes, &localTools); err != nil {
						return 1, fmt.Errorf("failed to unmarshal local tool: %w", err)
					}
				}
			}

			// tools (MCP)
			var allMCPTools mcpConnectionsAndTools = nil // key: streamable http url, value: tools
			if len(p.MCPTools.MCPStreamableURLs) > 0 {
				for _, serverURL := range p.MCPTools.MCPStreamableURLs {
					output.verbose(
						verboseMedium,
						p.Verbose,
						"fetching tools from '%s'...",
						stripURLParams(serverURL),
					)

					// connect,
					var mc *mcp.ClientSession
					if mc, err = mcpConnect(context.TODO(), serverURL); err == nil {
						// fetch tools,
						var fetchedTools []*mcp.Tool
						if fetchedTools, err = fetchMCPTools(
							context.TODO(),
							mc,
						); err == nil {
							if allMCPTools == nil {
								allMCPTools = mcpConnectionsAndTools{}
							}
							allMCPTools[serverURL] = struct {
								connection *mcp.ClientSession
								tools      []*mcp.Tool
							}{
								connection: mc,
								tools:      fetchedTools,
							}

							// check if there is any duplicated name of function
							if value, duplicated := duplicated(
								keysFromTools(localTools, allMCPTools),
							); duplicated {
								return 1, fmt.Errorf(
									"duplicated function name in tools: '%s'",
									value,
								)
							}
						} else {
							return 1, fmt.Errorf(
								"failed to fetch tools from '%s': %w",
								stripURLParams(serverURL),
								err,
							)
						}
					} else {
						return 1, fmt.Errorf(
							"failed to connect to MCP server '%s': %w",
							stripURLParams(serverURL),
							err,
						)
					}
				}
			}

			// close all MCP connections
			defer func() {
				for _, connsAndTools := range allMCPTools {
					_ = connsAndTools.connection.Close()
				}
			}()

			return doGeneration(
				context.TODO(),
				output,
				conf,
				*p.Model,
				*p.Generation.SystemInstruction,
				p.Generation.Temperature,
				p.Generation.TopP,
				p.Generation.TopK,
				p.Generation.Stop,
				p.Generation.OutputJSONScheme,
				p.Generation.WithThinking,
				p.Generation.HideReasoning,
				p.ContextWindowSize,
				*p.Generation.Prompt,
				p.Generation.Filepaths,
				p.Tools.ShowCallbackResults,
				p.Tools.RecurseOnCallbackResults,
				p.Tools.ForceCallDestructiveTools,
				localTools,
				p.LocalTools.ToolCallbacks,
				p.LocalTools.ToolCallbacksConfirm,
				allMCPTools,
				nil,
				p.UserAgent,
				p.ReplaceHTTPURLsInPrompt,
				p.Verbose,
			)
		}
	} else if p.ListModels {
		return doListModels(
			context.TODO(),
			output,
			conf,
			p,
		)
	} else { // otherwise,
		output.verbose(
			verboseMaximum,
			p.Verbose,
			"falling back with params: %s\n\n",
			prettify(p),
		)

		output.error("Parameter error: no task was requested or handled properly.")

		return output.printHelpBeforeExit(1, parser), nil
	}

	// should not reach here
}

// generate a default system instruction with given params
func defaultSystemInstruction(p params) string {
	datetime := time.Now().Format("2006-01-02 15:04:05 MST (Mon)")
	hostname, _ := os.Hostname()

	return fmt.Sprintf(defaultSystemInstructionFormat,
		appName,
		*p.Model,
		datetime,
		hostname,
	)
}
