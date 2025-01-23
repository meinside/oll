// ollama.go
//
// things for using Ollama APIs

package main

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"time"

	"github.com/ollama/ollama/api"
)

const (
	defaultModel = "deepseek-r1"
)

// generation parameter constants
const (
	defaultGenerationTemperature = float32(1.0)
	defaultGenerationTopP        = float32(0.95)
	defaultGenerationTopK        = int32(20)
)

// generate text with given things
func doGeneration(ctx context.Context, conf config, p params) (exit int, e error) {
	vbs := p.Verbose

	logVerbose(verboseMedium, vbs, "generating...")

	ctx, cancel := context.WithTimeout(ctx, time.Duration(conf.TimeoutSeconds)*time.Second)
	defer cancel()

	// ollama api client
	client, err := api.ClientFromEnvironment()
	if err != nil {
		return 1, err
	}

	prompt := *p.Prompt
	filesInPrompt := map[string][]byte{}
	if p.ReplaceHTTPURLsInPrompt {
		prompt, filesInPrompt = replaceURLsInPrompt(conf, p)

		logVerbose(verboseMedium, vbs, "prompt with urls replaced: '%s'", prompt)
	}

	// convert prompt with/without files
	prompt, imageFiles, err := convertPromptAndFiles(prompt, filesInPrompt, p.Filepaths)
	if err != nil {
		return 1, fmt.Errorf("failed to convert prompt and files: %w", err)
	}
	var images []api.ImageData = nil
	for _, img := range imageFiles {
		images = append(images, api.ImageData(img))
	}

	logVerbose(verboseMaximum, vbs, "with converted prompt: '%s' and image files: %v", strings.TrimSpace(prompt), imageFiles)

	// generation options
	req := &api.GenerateRequest{
		Model:  *p.Model,
		System: *p.SystemInstruction,
		Prompt: prompt,
		Images: images,
	}
	generationTemperature := defaultGenerationTemperature
	if p.Temperature != nil {
		generationTemperature = *p.Temperature
	}
	generationTopP := defaultGenerationTopP
	if p.TopP != nil {
		generationTopP = *p.TopP
	}
	generationTopK := defaultGenerationTopK
	if p.TopK != nil {
		generationTopK = *p.TopK
	}
	req.Options = map[string]any{
		"temperature": generationTemperature,
		"top_p":       generationTopP,
		"top_k":       generationTopK,
	}
	if p.OutputJSONScheme != nil {
		if json.Valid([]byte(*p.OutputJSONScheme)) {
			req.Format = json.RawMessage(*p.OutputJSONScheme)
		} else {
			return 1, fmt.Errorf("invalid output JSON scheme: `%s`", *p.OutputJSONScheme)
		}
	}

	logVerbose(verboseMaximum, vbs, "with generation request: %v", prettify(req))

	// generate
	type result struct {
		exit int
		err  error
	}
	ch := make(chan result, 1)
	go func() {
		endsWithNewLine := false

		if err = client.Generate(
			ctx,
			req,
			func(resp api.GenerateResponse) error {
				if len(resp.Response) > 0 {
					fmt.Print(resp.Response)

					endsWithNewLine = strings.HasSuffix(resp.Response, "\n")
				}
				if resp.Done {
					if !endsWithNewLine {
						fmt.Println()
					}

					// print the number of tokens
					logVerbose(
						verboseMinimum,
						vbs,
						"%s done[%s], load: %v, total: %v, prompt eval: %.3f/s, eval: %3f/s",
						*p.Model,
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
	res := <-ch

	return res.exit, res.err
}
