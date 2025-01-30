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

	"github.com/dustin/go-humanize"
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

// return a newly created ollama api client
func newClient() (*api.Client, error) {
	client, err := api.ClientFromEnvironment()
	if err != nil {
		return nil, err
	}

	return client, nil
}

// generate text with given things
func doGeneration(ctx context.Context, conf config, p params) (exit int, e error) {
	vbs := p.Verbose

	logVerbose(verboseMedium, vbs, "generating...")

	ctx, cancel := context.WithTimeout(ctx, time.Duration(conf.TimeoutSeconds)*time.Second)
	defer cancel()

	// ollama api client
	client, err := newClient()
	if err != nil {
		return 1, fmt.Errorf("failed to initialize Ollama API client: %w", err)
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
	if len(p.Stop) > 0 {
		stopSequences := []string{}
		for _, stop := range p.Stop {
			stopSequences = append(stopSequences, *stop)
		}
		req.Options["stop"] = stopSequences
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
	select {
	case <-ctx.Done():
		return 1, fmt.Errorf("generation timed out: %w", ctx.Err())
	case res := <-ch:
		return res.exit, res.err
	}
}

// list models
func doListModels(ctx context.Context, conf config, p params) (exit int, e error) {
	vbs := p.Verbose

	logVerbose(verboseMedium, vbs, "listing models...")

	ctx, cancel := context.WithTimeout(ctx, time.Duration(conf.TimeoutSeconds)*time.Second)
	defer cancel()

	// ollama api client
	client, err := newClient()
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
		logMessage(verboseNone, "%24s\t%s\n----", "name", "size")

		for _, model := range models.Models {
			if len(vbs) > 0 {
				logMessage(verboseNone, "%24s\t%s\t%s", model.Name, humanize.Bytes(uint64(model.Size)), prettify(model.Details))
			} else {
				logMessage(verboseNone, "%24s\t%s", model.Name, humanize.Bytes(uint64(model.Size)))
			}
		}
	} else {
		logMessage(verboseMedium, "no local models were found.")
	}

	return 0, nil
}

// generate embeddings
func doEmbeddings(ctx context.Context, conf config, p params) (exit int, e error) {
	vbs := p.Verbose

	logVerbose(verboseMedium, vbs, "generating embeddings...")

	ctx, cancel := context.WithTimeout(ctx, time.Duration(conf.TimeoutSeconds)*time.Second)
	defer cancel()

	// ollama api client
	client, err := newClient()
	if err != nil {
		return 1, fmt.Errorf("failed to initialize Ollama API client: %w", err)
	}

	// generate embeddings
	embeddings, err := client.Embeddings(ctx, &api.EmbeddingRequest{
		Model:  *p.Model,
		Prompt: *p.Prompt,
	})
	if err != nil {
		return 1, fmt.Errorf("failed to generate embeddings: %w", err)
	}

	// print floats
	floats, err := json.Marshal(embeddings.Embedding)
	if err != nil {
		return 1, fmt.Errorf("failed to marshal embeddings: %w", err)
	}
	logMessage(verboseNone, "%s", string(floats))

	return 0, nil
}
