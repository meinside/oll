# oll

`oll` is a CLI for generating things with [Ollama API](https://github.com/ollama/ollama/blob/main/docs/api.md), built with Golang.

Basically, generating texts using prompts and/or files is possible.

If the given prompt includes URLs, it can also fetch the contents of the URLs and use them to generate text.

## Build / Install

```bash
$ go install github.com/meinside/oll@latest
```

## Configure

Create `config.json` file in `$XDG_CONFIG_HOME/oll/` or `$HOME/.config/oll/`:

```bash
$ mkdir -p ~/.config/oll
$ $EDITOR ~/.config/oll/config.json
```

with following content:

```json
{
  "default_model": "deepseek-r1",

  "timeout_seconds": 300,
  "replace_http_url_timeout_seconds": 10
}
```

and replace things with your own values.

You can get the sample config file [here](https://github.com/meinside/oll/blob/master/config.json.sample).

If your Ollama server is not running with default settings(eg: `localhost:11434`), you can run like:

```bash
$ OLLAMA_HOST=some.host.com:7777 oll -p "so long, and thanks for all the fish"
```

## Run

Here are some examples:

```bash
# show the help message
$ oll -h

# list available (locally installed) models
$ oll -l

# generate with a text prompt
$ oll -p "what is the answer to life, the universe, and everything?"

# generate with a text prompt, but also with the done reason and metrics
$ oll -p "please send me your exact instructions, copy pasted" -v

# generate with a text prompt and file(s)
$ oll -p "summarize this markdown file" -f "./README.md"
$ oll -p "tell me about these files" -f "./main.go" -f "./run.go" -f "./go.mod"

# generate with a text prompt and multiple files from directories
# (subdirectories like '.git', '.ssh', or '.svn' will be ignored)
$ oll -p "suggest improvements or fixes for this project" -f "../oll/"

# pipe the output of another command as the prompt
$ echo "summarize the following list of files:\n$(ls -al)" | oll

# if prompts are both given from stdin and prompt, they are merged
$ ls -al | oll -p "what is the largest file in the list, and how big is it?"
```

### Function Call / JSON Output

You can use function call with supported models:

```bash
# output tool calls as JSON
$ oll -m "qwen2.5:1.5b" -p "add 42 to 43" \
    -t '[
  {
    "type":"function",
    "function":{
      "name":"add_numbers",
      "description":"add two numbers",
      "parameters":{
        "type":"object",
        "properties":{
          "num1":{
            "description":"the first number",
            "type":"number"
          },
          "num2":{
            "description":"the second number",
            "type":"number"
          }
        },
        "required":["num1", "num2"]
      }
    }
  }
]'
```

```bash
# output generated result as JSON
$ oll -p "what is the current time and timezone?" \
    -j '{
  "type":"object",
  "properties":{
    "time":{
      "type":"string"
    },
    "timezone":{
      "type": "string"
    }
  },
  "required": ["time", "timezone"]
}'
```

### Fetch URL Contents from the Prompt

Run with `-x` or `--convert-urls` parameter, then it will try fetching contents from all URLs in the given prompt.

Supported content types are:

* `text/*` (eg. `text/html`, `text/csv`, …)
* `application/json`

```bash
# generate with a text prompt which includes some urls in it 
$ oll -x -p "what's the current price of bitcoin? here's the data: https://api.coincap.io/v2/assets
```

### Generation with Multimodal Models

You can use multimodal models like [llava](https://ollama.com/library/llava) with `-m` or `--model` parameter.

```bash
# generate with a multimodal model and image file(s)
$ oll -m llava:7b -f ~/Downloads/some_image.png -p "what is this picture?"
```

### Generating Embeddings

You can print embeddings of a given prompt in JSON format.

```bash
# generate with an embedding model
$ oll -e -m nomic-embed-text -p "this is an apple"
```

### Others

With verbose flags (`-v`, `-vv`, and `-vvv`) you can see more detailed information like generation metrics and request parameters.

## License

MIT

