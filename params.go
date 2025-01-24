// params.go
//
// input parameters from command line and their helper functions

package main

// parameter definitions
type params struct {
	// config file's path
	ConfigFilepath *string `short:"c" long:"config" description:"Config file's path (default: $XDG_CONFIG_HOME/oll/config.json)"`

	// for ollama model
	Model             *string  `short:"m" long:"model" description:"Model to use (can be omitted)"`
	SystemInstruction *string  `short:"s" long:"system" description:"System instruction (can be omitted)"`
	Temperature       *float32 `long:"temperature" description:"'temperature' for generation (default: 1.0)"`
	TopP              *float32 `long:"top-p" description:"'top_p' for generation (default: 0.95)"`
	TopK              *int32   `long:"top-k" description:"'top_k' for generation (default: 20)"`

	// prompt and filepaths for generation
	Prompt    *string   `short:"p" long:"prompt" description:"Prompt for generation (can also be read from stdin)"`
	Filepaths []*string `short:"f" long:"filepath" description:"Path of a file or directory (can be used multiple times)"`

	// list models
	ListModels bool `short:"l" long:"list-models" description:"List available models (locally installed)"`

	// embedding
	GenerateEmbeddings bool `short:"e" long:"gen-embeddings" description:"Generate embeddings of the prompt"`

	// for fetching contents
	ReplaceHTTPURLsInPrompt bool    `short:"x" long:"convert-urls" description:"Convert URLs in the prompt to their text representations"`
	UserAgent               *string `long:"user-agent" description:"Override user-agent when fetching contents from URLs in the prompt"`

	// other options
	OutputJSONScheme *string `short:"j" long:"json" description:"Output result as this JSON scheme"`
	Verbose          []bool  `short:"v" long:"verbose" description:"Show verbose logs (can be used multiple times)"`
}

// check if prompt is given in the params
func (p *params) hasPrompt() bool {
	return p.Prompt != nil && len(*p.Prompt) > 0
}

// check if any task is requested
// FIXME: TODO: need to be fixed whenever a new task is added
func (p *params) taskRequested() bool {
	return p.hasPrompt() || p.ListModels || p.GenerateEmbeddings
}

// check if multiple tasks are requested
// FIXME: TODO: need to be fixed whenever a new task is added
func (p *params) multipleTaskRequested() bool {
	hasPrompt := p.hasPrompt()
	promptCounted := false
	num := 0

	if p.ListModels { // list locally installed models
		num++
		if hasPrompt && !promptCounted {
			num++
			promptCounted = true
		}
	}
	if p.GenerateEmbeddings { // generate embeddings
		num++
		if hasPrompt && !promptCounted {
			promptCounted = true
		}
	}
	// TODO: add conditions for other tasks

	if hasPrompt && !promptCounted { // no other tasks requested, but prompt is given
		num++
	}

	return num > 1
}
