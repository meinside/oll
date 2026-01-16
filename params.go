// params.go
//
// input parameters from command line and their helper functions

package main

// parameter definitions
type params struct {
	// for showing the version
	ShowVersion bool `long:"version" description:"Show the version of this application"`

	// config file's path
	ConfigFilepath *string `short:"c" long:"config" description:"Config file's path (default: $XDG_CONFIG_HOME/oll/config.json)"`

	// for ollama model
	Model                   *string `short:"m" long:"model" description:"Model to use (can be omitted)"`
	ModelForImageGeneration *string `long:"image-generation-model" description:"Model for image generation (can be omitted)"`

	// parameters for generation
	//
	// https://github.com/ollama/ollama/blob/main/docs/api.md#generate-a-chat-completion
	Generation struct {
		// prompt, system instruction, and other things for generation
		Prompt            *string   `short:"p" long:"prompt" description:"Prompt for generation (can also be read from stdin)"`
		Filepaths         []*string `short:"f" long:"filepath" description:"Path of a file or directory (can be used multiple times)"`
		SystemInstruction *string   `short:"s" long:"system" description:"System instruction (can be omitted)"`
		Temperature       *float32  `long:"temperature" description:"'temperature' for generation (default: 1.0)"`
		TopP              *float32  `long:"top-p" description:"'top_p' for generation (default: 0.95)"`
		TopK              *int32    `long:"top-k" description:"'top_k' for generation (default: 20)"`
		Stop              []*string `long:"stop" description:"'stop' sequence string for generation (can be used multiple times)"`

		// other generation options
		OutputJSONScheme *string `short:"j" long:"json" description:"Output result as this JSON scheme"`

		// thinking
		WithThinking  bool `short:"T" long:"with-thinking" description:"Generate with thinking (works only with models which support thinking)"`
		HideReasoning bool `short:"H" long:"hide-reasoning" description:"Hide reasoning (<think></think>) while streaming the result"`

		// image generation
		WithImages             bool    `short:"I" long:"with-images" description:"Generate images with this prompt (works only with models which support image generation)"`
		NegativePrompt         *string `long:"negative-prompt" description:"Negative prompt for image generation"`
		ImageWidth             *int    `long:"image-width" description:"Width for image generation"`
		ImageHeight            *int    `long:"image-height" description:"Height for image generation"`
		SaveImagesToDir        *string `long:"save-images-to-dir" description:"Save generated images to this directory (default: $TMPDIR)"`
		DisplayImageInTerminal bool    `long:"display-image-in-terminal" description:"Display generated images in terminal"`
	} `group:"Generation"`

	// tools
	Tools struct {
		ShowCallbackResults      bool `long:"show-callback-results" description:"Whether to force print the results of tool callbacks (default: only in verbose mode)"`
		RecurseOnCallbackResults bool `short:"r" long:"recurse-on-callback-results" description:"Whether to do recursive generations on callback results (default: false)"`

		ForceCallDestructiveTools bool `long:"force-call-destructive-tools" description:"Whether to force calling destructive tools without asking"`
	} `group:"Tools"`

	// tools (local)
	LocalTools struct {
		Tools                *string           `short:"t" long:"tools" description:"Tools for function call (in JSON)"`
		ToolCallbacks        map[string]string `long:"tool-callbacks" description:"Tool callbacks (can be used multiple times, eg. 'fn_name1:/path/to/script1.sh', 'fn_name2:/path/to/script2.sh')"`
		ToolCallbacksConfirm map[string]bool   `long:"tool-callbacks-confirm" description:"Confirm before executing tool callbacks (can be used multiple times, eg. 'fn_name1:true', 'fn_name2:false')"`
	} `group:"Tools (Local)"`

	// tools (MCP)
	MCPTools struct {
		StreamableURLs []string `long:"mcp-streamable-url" description:"Streamable URL of MCP server for function call (can be used multiple times)"`
		StdioCommands  []string `long:"mcp-stdio-command" description:"Commands of local stdio MCP Tools (can be used multiple times)"`
	} `group:"Tools (MCP)"`

	// list models
	//
	// https://github.com/ollama/ollama/blob/main/docs/api.md#list-local-models
	ListModels bool `short:"l" long:"list-models" description:"List available models (locally installed)"`

	// embedding
	//
	// https://github.com/ollama/ollama/blob/main/docs/api.md#generate-embeddings
	Embeddings struct {
		GenerateEmbeddings            bool  `short:"e" long:"gen-embeddings" description:"Generate embeddings of the prompt"`
		EmbeddingsChunkSize           *uint `long:"embeddings-chunk-size" description:"Chunk size for embeddings (default: 4096)"`
		EmbeddingsOverlappedChunkSize *uint `long:"embeddings-overlapped-chunk-size" description:"Overlapped size of chunks for embeddings (default: 64)"`
	} `group:"Embeddings"`

	// for fetching contents
	ReplaceHTTPURLsInPrompt bool    `short:"x" long:"convert-urls" description:"Convert URLs in the prompt to their text representations"`
	UserAgent               *string `long:"user-agent" description:"Override user-agent when fetching contents from URLs in the prompt"`

	// https://github.com/ollama/ollama/blob/main/docs/faq.md#how-can-i-specify-the-context-window-size
	ContextWindowSize *int `short:"w" long:"context-window-size" description:"Context window size of the prompt (default: 2048)"`

	// other options
	Verbose []bool `short:"v" long:"verbose" description:"Show verbose logs (can be used multiple times)"`
}

// hasPrompt checks if prompt is given in the params.
func (p *params) hasPrompt() bool {
	return p.Generation.Prompt != nil && len(*p.Generation.Prompt) > 0
}

// taskRequested checks if any task is requested.
//
// FIXME: TODO: need to be fixed whenever a new task is added
func (p *params) taskRequested() bool {
	return p.hasPrompt() ||
		p.ListModels ||
		p.Embeddings.GenerateEmbeddings ||
		p.ShowVersion
}

// multipleTaskRequested checks if multiple tasks are requested.
//
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
	if p.Embeddings.GenerateEmbeddings { // generate embeddings
		num++
		if hasPrompt && !promptCounted {
			promptCounted = true
		}
	}
	if p.ShowVersion { // show version
		num++
		if hasPrompt && !promptCounted {
			num++
			promptCounted = true
		}
	}
	// TODO: add conditions for other tasks

	if hasPrompt && !promptCounted { // no other tasks requested, but prompt is given
		num++
	}

	return num > 1
}
