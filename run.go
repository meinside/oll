// run.go

package main

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"time"

	"github.com/jessevdk/go-flags"
	"github.com/meinside/smithery-go"
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
func run(parser *flags.Parser, p params) (exitCode int, err error) {
	// early return if no task was requested
	if !p.taskRequested() {
		logMessage(
			verboseMedium,
			"No task was requested.",
		)

		return printHelpBeforeExit(1, parser), nil
	}

	// early return after printing the version
	if p.ShowVersion {
		logMessage(
			verboseMinimum,
			"%s %s\n\n",
			appName,
			version.Build(version.OS|version.Architecture),
		)

		return printHelpBeforeExit(0, parser), nil
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
	p.Generation.Filepaths, err = expandFilepaths(p)
	if err != nil {
		return 1, fmt.Errorf("failed to read given filepaths: %w", err)
	}

	if p.hasPrompt() { // if prompt is given,
		if p.Embeddings.GenerateEmbeddings {
			logVerbose(
				verboseMaximum,
				p.Verbose,
				"embeddings request params with prompt: %s\n\n",
				prettify(p),
			)

			return doEmbeddingsGeneration(context.TODO(), conf, p)
		} else {
			logVerbose(
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

			// tools (smithery)
			var sc *smithery.Client
			var allSmitheryTools map[string][]*mcp.Tool = nil
			if conf.SmitheryAPIKey != nil &&
				p.SmitheryTools.SmitheryProfileID != nil &&
				len(p.SmitheryTools.SmitheryServerNames) > 0 {
				sc = newSmitheryClient(*conf.SmitheryAPIKey)

				for _, smitheryServerName := range p.SmitheryTools.SmitheryServerNames {
					logVerbose(
						verboseMedium,
						p.Verbose,
						"fetching tools for '%s' from smithery...",
						smitheryServerName,
					)

					var fetchedSmitheryTools []*mcp.Tool
					if fetchedSmitheryTools, err = fetchSmitheryTools(
						context.TODO(),
						sc,
						*p.SmitheryTools.SmitheryProfileID,
						smitheryServerName,
					); err == nil {
						if allSmitheryTools == nil {
							allSmitheryTools = map[string][]*mcp.Tool{}
						}
						allSmitheryTools[smitheryServerName] = fetchedSmitheryTools

						// check if there is any duplicated name of function
						if value, duplicated := duplicated(
							keysFromTools(localTools, allSmitheryTools),
						); duplicated {
							return 1, fmt.Errorf(
								"duplicated function name in tools: '%s'",
								value,
							)
						}
					} else {
						return 1, fmt.Errorf(
							"failed to fetch tools from smithery: %w",
							err,
						)
					}
				}
			} else if p.SmitheryTools.SmitheryProfileID != nil || len(p.SmitheryTools.SmitheryServerNames) > 0 {
				if conf.SmitheryAPIKey == nil {
					warn(
						"Smithery API key is not set in the config file, so ignoring it for now.",
					)
				} else {
					warn(
						"Both profile id and server name is needed for using Smithery, so ignoring them for now.",
					)
				}
			}

			return doGeneration(
				context.TODO(),
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
				localTools,
				p.LocalTools.ToolCallbacks,
				p.LocalTools.ToolCallbacksConfirm,
				conf.SmitheryAPIKey,
				sc,
				p.SmitheryTools.SmitheryProfileID,
				allSmitheryTools,
				nil,
				p.UserAgent,
				p.ReplaceHTTPURLsInPrompt,
				p.Verbose,
			)
		}
	} else if p.ListModels {
		return doListModels(context.TODO(), conf, p)
	} else { // otherwise,
		logVerbose(
			verboseMaximum,
			p.Verbose,
			"falling back with params: %s\n\n",
			prettify(p),
		)

		logMessage(
			verboseMedium,
			"Parameter error: no task was requested or handled properly.",
		)

		return printHelpBeforeExit(1, parser), nil
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
