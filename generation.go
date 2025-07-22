// generation.go
//
// things for generation with Ollama APIs

package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"slices"
	"strings"
	"text/template"
	"time"

	"github.com/dustin/go-humanize"
	"github.com/fatih/color"
	"github.com/ollama/ollama/api"
)

const (
	// https://ollama.com/library/mistral-small3.2
	defaultModel = "mistral-small3.2:24b" // NOTE: picked a model which supports both tooling and vision for convenience
)

// generation parameter constants
const (
	defaultGenerationTemperature = float32(1.0)
	defaultGenerationTopP        = float32(0.95)
	defaultGenerationTopK        = int32(20)

	defaultEmbeddingsChunkSize           uint = 2048 * 2
	defaultEmbeddingsChunkOverlappedSize uint = 64
)

// return a newly created ollama api client
func newOllamaClient() (*api.Client, error) {
	client, err := api.ClientFromEnvironment()
	if err != nil {
		return nil, err
	}

	return client, nil
}

// generate text with given things
//
// https://github.com/ollama/ollama/blob/main/docs/api.md#generate-a-chat-completion
func doGeneration(
	ctx context.Context,
	output *outputWriter,
	conf config,
	model string,
	systemInstruction string,
	temperature *float32,
	topP *float32,
	topK *int32,
	stop []*string,
	outputJSONScheme *string,
	withThinking, hideReasoning bool,
	contextWindowSize *int,
	prompt string,
	filepaths []*string,
	showCallbackResults, recurseOnCallbackResults bool, forceCallDestructiveTools bool,
	localTools []api.Tool,
	localToolCallbacks map[string]string,
	localToolCallbacksConfirm map[string]bool,
	mcpConnsAndTools mcpConnectionsAndTools,
	pastGenerations []api.Message,
	userAgent *string,
	replaceHTTPURLsInPrompt bool,
	vbs []bool,
) (exit int, e error) {
	output.verbose(
		verboseMedium,
		vbs,
		"generating...",
	)

	ctx, cancel := context.WithTimeout(
		ctx,
		time.Duration(conf.TimeoutSeconds)*time.Second,
	)
	defer cancel()

	// ollama api client
	client, err := newOllamaClient()
	if err != nil {
		return 1, fmt.Errorf("failed to initialize Ollama API client: %w", err)
	}

	filesInPrompt := map[string][]byte{}
	if replaceHTTPURLsInPrompt {
		prompt, filesInPrompt = replaceURLsInPrompt(
			output,
			conf,
			userAgent,
			prompt,
			vbs,
		)

		output.verbose(
			verboseMedium,
			vbs,
			"prompt with urls replaced: '%s'",
			prompt,
		)
	}

	// convert prompt with/without files
	prompt, imageFiles, err := convertPromptAndFiles(
		prompt,
		filesInPrompt,
		filepaths,
	)
	if err != nil {
		return 1, fmt.Errorf("failed to convert prompt and files: %w", err)
	}
	var images []api.ImageData = nil
	for _, img := range imageFiles {
		images = append(images, api.ImageData(img))
	}

	// TODO: return error when the context length is exceeded

	output.verbose(
		verboseMaximum,
		vbs,
		"with converted prompt: '%s' and image files: %v",
		strings.TrimSpace(prompt),
		imageFiles,
	)

	// generation options
	req := &api.ChatRequest{
		Model: model,
		Messages: []api.Message{
			{
				Role:    "system",
				Content: systemInstruction,
			},
			{
				Role:    "user",
				Content: prompt,
				Images:  images,
			},
		},
	}
	generationTemperature := defaultGenerationTemperature
	if temperature != nil {
		generationTemperature = *temperature
	}
	generationTopP := defaultGenerationTopP
	if topP != nil {
		generationTopP = *topP
	}
	generationTopK := defaultGenerationTopK
	if topK != nil {
		generationTopK = *topK
	}
	req.Options = map[string]any{
		"temperature": generationTemperature,
		"top_p":       generationTopP,
		"top_k":       generationTopK,
	}
	if contextWindowSize != nil {
		req.Options["num_ctx"] = *contextWindowSize
	}
	if len(stop) > 0 {
		stopSequences := []string{}
		for _, stop := range stop {
			stopSequences = append(stopSequences, *stop)
		}
		req.Options["stop"] = stopSequences
	}
	if outputJSONScheme != nil {
		if json.Valid([]byte(*outputJSONScheme)) {
			req.Format = json.RawMessage(*outputJSONScheme)
		} else {
			return 1, fmt.Errorf("invalid output JSON scheme: `%s`", *outputJSONScheme)
		}
	}
	// (tools - local)
	if len(localTools) > 0 {
		req.Tools = append(req.Tools, localTools...)
	}
	// (tools - MCP)
	var ollamaTools []api.Tool = nil
	for _, connsAndTools := range mcpConnsAndTools {
		if converted, err := mcpToOllamaTools(connsAndTools.tools); err == nil {
			for _, c := range converted {
				ollamaTools = append(ollamaTools, *c)
			}
		} else {
			return 1, fmt.Errorf("failed to convert MCP tools: %w", err)
		}
	}
	if len(ollamaTools) > 0 {
		req.Tools = append(req.Tools, ollamaTools...)
	}
	// (thinking)
	req.Think = ptr(withThinking)

	// (history)
	req.Messages = append(req.Messages, pastGenerations...)

	output.verbose(
		verboseMaximum,
		vbs,
		"with generation request: %v",
		prettify(req),
	)

	// generate
	type result struct {
		exit int
		err  error
	}
	ch := make(chan result, 1)
	go func() {
		reasoningStarted := false
		firstContentAfterReasoning := false

		if err = client.Chat(
			ctx,
			req,
			func(resp api.ChatResponse) error {
				if resp.Message.Role == "assistant" {
					// handle the beginning and end of reasoning
					if len(resp.Message.Thinking) > 0 {
						if !reasoningStarted {
							if !hideReasoning {
								// print generated content
								output.printColored(
									color.FgHiGreen,
									"<think>\n",
								)
								pastGenerations = appendModelResponseToPastGenerations(
									pastGenerations,
									"<think>\n",
								)
							}

							reasoningStarted = true
						}
					} else {
						if reasoningStarted {
							if !hideReasoning {
								// print generated content
								output.printColored(
									color.FgHiGreen,
									"</think>\n",
								)
								pastGenerations = appendModelResponseToPastGenerations(
									pastGenerations,
									"</think>\n",
								)
							}

							reasoningStarted = false
							firstContentAfterReasoning = true
						}
					}

					// show thinking
					if !hideReasoning && len(resp.Message.Thinking) > 0 {
						// print generated content
						output.printColored(
							color.FgHiWhite,
							"%s",
							resp.Message.Thinking,
						)
						pastGenerations = appendModelResponseToPastGenerations(
							pastGenerations,
							resp.Message.Thinking,
						)
					}

					// handle generated things
					if len(resp.Message.Content) > 0 {
						// print the generated content
						if !hideReasoning || !reasoningStarted {
							content := resp.Message.Content

							// trim the first content after reasoning for removing unwanted newlines
							if hideReasoning && firstContentAfterReasoning {
								content = strings.TrimSpace(content)
								firstContentAfterReasoning = false
							}

							// print generated content
							output.printColored(
								color.FgHiWhite,
								"%s",
								content,
							)
							pastGenerations = appendModelResponseToPastGenerations(
								pastGenerations,
								content,
							)
						}
					} else if len(resp.Message.ToolCalls) > 0 {
						marshalled, _ := json.MarshalIndent(
							resp.Message.ToolCalls,
							"",
							"  ",
						)

						output.verbose(
							verboseMedium,
							vbs,
							"generated tool calls: %s",
							string(marshalled),
						)

						// call functions
						for _, call := range resp.Message.ToolCalls {
							output.verbose(
								verboseMedium,
								vbs,
								"calling callback for tool call: %s",
								call.Function.Name,
							)

							fn := fmt.Sprintf(
								"%s(%s)",
								call.Function.Name,
								prettify(call.Function.Arguments, true),
							)

							if callbackPath, exists := localToolCallbacks[call.Function.Name]; exists {
								// with local tools,
								fnCallback, okToRun := checkCallbackPath(
									output,
									callbackPath,
									localToolCallbacksConfirm,
									forceCallDestructiveTools,
									call.Function,
									vbs,
								)

								if okToRun {
									output.verbose(
										verboseMedium,
										vbs,
										"executing callback...",
									)

									if res, err := fnCallback(); err != nil {
										return fmt.Errorf(
											"tool callback failed: %s",
											err,
										)
									} else {
										// warn that there are ignored tool callbacks
										if len(localToolCallbacks) > 0 &&
											!recurseOnCallbackResults {
											output.warn(
												"Not recursing, ignoring the result of '%s'.",
												fn,
											)
										}

										// print the result of execution
										if showCallbackResults ||
											verboseLevel(vbs) >= verboseMinimum {
											output.printColored(
												color.FgHiCyan,
												"%s\n",
												res,
											)
										}

										// append function call result
										pastGenerations = append(pastGenerations, api.Message{
											Role: "user",
											Content: fmt.Sprintf(
												`Result of function '%s':

%s`,
												fn,
												res,
											),
										})
									}
								} else {
									skipped := fmt.Sprintf(
										"Skipped execution of callback '%s' for function '%s'.\n",
										callbackPath,
										fn,
									)

									// print generated content
									output.printColored(
										color.FgHiWhite,
										"%s",
										skipped,
									)
									pastGenerations = appendUserMessageToPastGenerations(
										pastGenerations,
										skipped,
									)

									// append function call result (not called)
									pastGenerations = append(pastGenerations, api.Message{
										Role: "user",
										Content: fmt.Sprintf(
											`User chose not to call function '%s'.`,
											fn,
										),
									})
								}
							} else if serverURL, mc, tool, exists := mcpToolFrom(
								mcpConnsAndTools,
								call.Function.Name,
							); exists {
								// NOTE: avoid infinite loops
								if slices.ContainsFunc(pastGenerations, func(message api.Message) bool {
									return strings.Contains(message.Content, fn)
								}) {
									return fmt.Errorf("possible infinite loop detected: '%s'", fn)
								}

								okToRun := false

								// check if matched MCP tool requires confirmation
								if tool.Annotations != nil &&
									tool.Annotations.DestructiveHint != nil &&
									*tool.Annotations.DestructiveHint &&
									!forceCallDestructiveTools {
									okToRun = confirm(fmt.Sprintf(
										"May I call tool '%s' from '%s' for function '%s'?",
										call.Function.Name,
										stripURLParams(serverURL),
										fn,
									))
								} else {
									okToRun = true
								}

								if okToRun {
									if res, err := fetchToolCallResult(
										ctx,
										mc,
										call.Function.Name,
										call.Function.Arguments,
									); err == nil {
										fnResult := fmt.Sprintf(
											"Tool call result of '%s':\n%s",
											fn,
											prettify(res.Content),
										)

										// print the result of execution
										if showCallbackResults ||
											verboseLevel(vbs) >= verboseMinimum {
											output.printColored(
												color.FgHiCyan,
												"%s\n",
												fnResult,
											)
										}

										// print generated content
										pastGenerations = appendUserMessageToPastGenerations(
											pastGenerations,
											fnResult,
										)
									} else {
										return fmt.Errorf("failed to call MCP tool: %w", err)
									}
								} else {
									output.printColored(
										color.FgHiYellow,
										"Skipped execution of MCP tool '%s' from '%s' for function '%s'.\n",
										call.Function.Name,
										stripURLParams(serverURL),
										fn,
									)

									// append function call result (not called)
									pastGenerations = appendUserMessageToPastGenerations(
										pastGenerations,
										fmt.Sprintf(
											`User chose not to call function '%s'.`,
											fn,
										),
									)
								}
							} else {
								// print generated content
								fnUnhandled := fmt.Sprintf("Generated tool call: %s", fn)
								output.printColored(
									color.FgHiWhite,
									"%s\n",
									fnUnhandled,
								)
								pastGenerations = appendUserMessageToPastGenerations(
									pastGenerations,
									fnUnhandled,
								)
							}
						}
					} else if len(resp.Message.Images) > 0 {
						output.verbose(
							verboseMedium,
							vbs,
							"generated %d images",
							len(resp.Message.Images),
						)

						// TODO: handle images
						handled := fmt.Sprintf("Generated %d images.", len(resp.Message.Images))
						// FIXME: print generated content
						output.printColored(
							color.FgHiWhite,
							"%s\n",
							handled,
						)
						pastGenerations = appendModelResponseToPastGenerations(
							pastGenerations,
							handled,
						)
					}
				}
				if resp.Done {
					output.makeSureToEndWithNewLine()

					// print the number of tokens
					output.verbose(
						verboseMinimum,
						vbs,
						"%s done[%s], load: %v, total: %v, prompt eval: %.3f/s, eval: %3f/s",
						model,
						resp.DoneReason,
						resp.LoadDuration,
						resp.TotalDuration,
						float64(resp.PromptEvalCount)/resp.PromptEvalDuration.Seconds(),
						float64(resp.EvalCount)/resp.EvalDuration.Seconds(),
					)

					// success
					ch <- result{
						exit: 0,
						err:  nil,
					}
				}

				return nil
			},
		); err != nil {
			// error
			ch <- result{
				exit: 1,
				err:  fmt.Errorf("generation failed: %w", err),
			}
		}
	}()

	// wait for the generation to finish
	select {
	case <-ctx.Done():
		return 1, fmt.Errorf("generation timed out: %w", ctx.Err())
	case res := <-ch:
		// check if recursion is needed
		if res.exit == 0 &&
			res.err == nil &&
			recurseOnCallbackResults &&
			historyEndsWithUsers(pastGenerations) {
			output.verbose(
				verboseMedium,
				vbs,
				"Generating recursively with history: %s",
				prettify(pastGenerations),
			)

			// recurse!
			return doGeneration(
				ctx,
				output,
				conf,
				model,
				systemInstruction,
				temperature,
				topP,
				topK,
				stop,
				outputJSONScheme,
				withThinking,
				hideReasoning,
				contextWindowSize,
				prompt,
				filepaths,
				showCallbackResults,
				recurseOnCallbackResults,
				forceCallDestructiveTools,
				localTools,
				localToolCallbacks,
				localToolCallbacksConfirm,
				mcpConnsAndTools,
				pastGenerations,
				userAgent,
				replaceHTTPURLsInPrompt,
				vbs,
			)
		}

		return res.exit, res.err
	}
}

// list models
//
// https://github.com/ollama/ollama/blob/main/docs/api.md#list-local-models
func doListModels(
	ctx context.Context,
	output *outputWriter,
	conf config,
	p params,
) (exit int, e error) {
	vbs := p.Verbose

	output.verbose(
		verboseMedium,
		vbs,
		"listing models...",
	)

	ctx, cancel := context.WithTimeout(
		ctx,
		time.Duration(conf.TimeoutSeconds)*time.Second,
	)
	defer cancel()

	// ollama api client
	client, err := newOllamaClient()
	if err != nil {
		return 1, fmt.Errorf("failed to initialize Ollama API client: %w", err)
	}

	// list models
	models, err := client.List(ctx)
	if err != nil {
		return 1, fmt.Errorf("failed to list models: %w", err)
	}
	if len(models.Models) > 0 {
		// print headers
		output.printColored(
			color.FgWhite,
			"%24s\t%s\n----\n",
			"name",
			"size",
		)

		for _, model := range models.Models {
			if len(vbs) > 0 {
				output.printColored(
					color.FgHiWhite,
					"%24s\t%s\t%s\n",
					model.Name,
					humanize.Bytes(uint64(model.Size)),
					prettify(model.Details),
				)
			} else {
				output.printColored(
					color.FgHiWhite,
					"%24s\t%s\n",
					model.Name,
					humanize.Bytes(uint64(model.Size)),
				)
			}
		}
	} else {
		output.printColored(
			color.FgHiRed,
			"no local models were found.",
		)
	}

	return 0, nil
}

// generate embeddings
//
// https://github.com/ollama/ollama/blob/main/docs/api.md#generate-embeddings
func doEmbeddingsGeneration(
	ctx context.Context,
	output *outputWriter,
	conf config,
	p params,
) (exit int, e error) {
	vbs := p.Verbose

	output.verbose(
		verboseMedium,
		vbs,
		"generating embeddings...",
	)

	if p.Embeddings.EmbeddingsChunkSize == nil {
		p.Embeddings.EmbeddingsChunkSize = ptr(defaultEmbeddingsChunkSize)
	}
	if p.Embeddings.EmbeddingsOverlappedChunkSize == nil {
		p.Embeddings.EmbeddingsOverlappedChunkSize = ptr(defaultEmbeddingsChunkOverlappedSize)
	}

	// chunk prompt text
	chunks, err := ChunkText(*p.Generation.Prompt, TextChunkOption{
		ChunkSize:      *p.Embeddings.EmbeddingsChunkSize,
		OverlappedSize: *p.Embeddings.EmbeddingsOverlappedChunkSize,
		EllipsesText:   "...",
	})
	if err != nil {
		return 1, fmt.Errorf("failed to chunk text: %w", err)
	}

	ctx, cancel := context.WithTimeout(
		ctx,
		time.Duration(conf.TimeoutSeconds)*time.Second,
	)
	defer cancel()

	// ollama api client
	client, err := newOllamaClient()
	if err != nil {
		return 1, fmt.Errorf("failed to initialize Ollama API client: %w", err)
	}

	options := map[string]any{}
	if p.ContextWindowSize != nil {
		options["num_ctx"] = *p.ContextWindowSize
	}

	// iterate chunks and generate embeddings
	type embedding struct {
		Text    string    `json:"text"`
		Vectors []float64 `json:"vectors"`
	}
	type embeddings struct {
		Original string      `json:"original"`
		Chunks   []embedding `json:"chunks"`
	}
	embeds := embeddings{
		Original: *p.Generation.Prompt,
		Chunks:   []embedding{},
	}
	for i, text := range chunks.Chunks {
		embeddings, err := client.Embeddings(ctx, &api.EmbeddingRequest{
			Model:   *p.Model,
			Prompt:  text,
			Options: options,
		})
		if err != nil {
			return 1, fmt.Errorf("embeddings failed for chunk[%d]: %w", i, err)
		} else {
			embeds.Chunks = append(embeds.Chunks, embedding{
				Text:    text,
				Vectors: embeddings.Embedding,
			})
		}
	}

	// print floats
	floats, err := json.Marshal(embeds)
	if err != nil {
		return 1, fmt.Errorf("failed to marshal embeddings: %w", err)
	}
	output.printColored(
		color.FgHiWhite,
		"%s\n",
		string(floats),
	)

	return 0, nil
}

// predefined callback function names
const (
	fnCallbackStdin     = `@stdin`
	fnCallbackFormatter = `@format`
)

// check if given `callbackPath` is executable
func checkCallbackPath(
	output *outputWriter,
	callbackPath string,
	confirmToolCallbacks map[string]bool,
	forceCallDestructiveTools bool,
	fnCall api.ToolCallFunction,
	vbs []bool,
) (
	fnCallback func() (string, error),
	okToRun bool,
) {
	// check if `callbackPath` is a predefined callback
	if callbackPath == fnCallbackStdin { // @stdin
		okToRun = true

		fnCallback = func() (string, error) {
			prompt := fmt.Sprintf(
				"Type your answer for function '%s(%s)'",
				fnCall.Name,
				prettify(fnCall.Arguments, true),
			)

			return readFromStdin(prompt)
		}
	} else if strings.HasPrefix(callbackPath, fnCallbackFormatter) { // @format
		okToRun = true

		fnCallback = func() (string, error) {
			if tpl, exists := strings.CutPrefix(callbackPath, fnCallbackFormatter+"="); exists {
				if t, err := template.New("fnFormatter").Parse(tpl); err == nil {
					buf := new(bytes.Buffer)
					if err := t.Execute(buf, fnCall.Arguments); err == nil {
						return buf.String(), nil
					} else {
						return "", fmt.Errorf("failed to execute template for %s: %w", fnCallbackFormatter, err)
					}
				} else {
					return "", fmt.Errorf("failed to parse format for %s: %w", fnCallbackFormatter, err)
				}
			} else {
				if marshalled, err := json.MarshalIndent(fnCall.Arguments, "", "  "); err == nil {
					return string(marshalled), nil
				} else {
					return "", fmt.Errorf("failed to marshal to JSON for %s: %w", fnCallbackFormatter, err)
				}
			}
		}
	} else { // ordinary path of binary/script:
		// ask for confirmation
		if confirmNeeded, exists := confirmToolCallbacks[fnCall.Name]; exists && confirmNeeded && !forceCallDestructiveTools {
			okToRun = confirm(fmt.Sprintf(
				"May I execute callback '%s' for function '%s(%s)'?",
				callbackPath,
				fnCall.Name,
				prettify(fnCall.Arguments, true),
			))
		} else {
			okToRun = true
		}

		// run executable
		fnCallback = func() (string, error) {
			output.verbose(
				verboseMinimum,
				vbs,
				"executing callback '%s' for function '%s(%s)'...",
				callbackPath,
				fnCall.Name,
				prettify(fnCall.Arguments, true),
			)

			return runExecutable(callbackPath, fnCall.Arguments)
		}
	}

	return
}
