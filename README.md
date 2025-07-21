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
  "default_model": "mistral-small3.2:24b",

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
$ curl https://api.coincap.io/v2/assets \
    | jq -c '[.data[] | {id, priceUsd}][0:10]' \
    | oll -p "what's the current price of bitcoin?"
```

You can generate with thinking with [models which support thinking](https://ollama.com/search?c=thinking):

```bash
$ oll -m "qwen3:8b" -p "what is the earth escape velocity?" --with-thinking
$ oll -m "qwen3:8b" -p "what is the three laws of newton?" --with-thinking --hide-reasoning
```

### JSON Output

Print generated result as JSON with `-j` or `--json`:

```bash
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

### Function Call (Local)

You can use function call with [supported models](https://ollama.com/search?c=tools):
(NOTE: some models may not support tools)

```bash
# output tool calls as JSON
$ oll -m "mistral-small3.2:24b" \
    -p "add 42 to 43" \
    --tools '[
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

#### Execute Callbacks on Function Calls

With `--tool-callbacks`, it will run matched scripts/binaries with the function call data.

Here is a sample bash script `categorize_image.sh` which categorizes given image with function call:

```bash
#!/usr/bin/env bash
#
# categorize_image.sh

CALLBACK_SCRIPT="/path/to/callback_categorize_image.sh"

# read filename from args
filename="$*"

# tools
read -r -d '' TOOLS <<-'EOF'
[
  {
    "type": "function",
    "function": {
      "name": "categorize_image",
      "description": "this function categorizes the provided image",
      "parameters": {
        "type": "OBJECT",
        "properties": {
          "category": {
            "type": "STRING",
            "description": "the category of the provided image",
            "enum": ["animal", "person", "scenary", "object", "other"],
            "nullable": false
          },
          "description": {
            "type": "STRING",
            "description": "the detailed description of the provided image",
            "nullable": false
          }
        },
        "required": ["category", "description"]
      }
    }
  }
]
EOF

# run oll with params (drop error/warning messages)
oll -m "mistral-small3.2:24b" \
    -p "categorize this image" \
    -f "$filename" \
    --tools="$TOOLS" \
    --tool-callbacks="categorize_image:$CALLBACK_SCRIPT" \
    --show-callback-results 2>/dev/null
```

And this is a callback script `callback_categorize_image.sh`:

```bash
#!/usr/bin/env bash
#
# callback_categorize_image.sh

# args (in JSON)
data="$*"

# read args with jq
result=$(echo "$data" |
  jq -r '. | "Category: \(.category)\nDescription: \(.description)"')

# print to stdout
echo "$result"
```

Run `categorize_image.sh` with an image file:

```bash
$ ./categorize_image.sh /path/to/some_image.jpg
```

then it will print the desired result:

```bash
Category: scenary
Description: a group of people walking on the street in a city
```

#### Confirm before Executing Callbacks

With `--tool-callbacks-confirm`, it will ask for confirmation before executing the scripts/binaries:

```bash
$ oll -m "mistral-small3.2:24b" \
    -p "nuke the tmp directory" \
    --tools='[{
        "type": "function",
        "function": {
            "name": "remove_dir_recursively",
            "description": "this function deletes given directory recursively", 
            "parameters": {
                "type": "OBJECT",
                "properties": {"directory": {"type": "STRING"}},
                "required": ["directory"]
            }
        }
    }]' \
    --tool-callbacks="remove_dir_recursively:/path/to/rm_rf_dir.sh" \
    --tool-callbacks-confirm="remove_dir_recursively:true" \
    --recurse-on-callback-results
```

#### Generate Recursively with Callback Results

With `--recurse-on-callback-results` / `-r`, it will generate recursively with the results of the scripts/binaries:

```bash
$ oll -m "mistral-small3.2:24b" \
    -p "what is the smallest .sh file in /home/ubuntu/tmp/ and how many lines does that file have" \
    --tools='[
    {
        "type": "function",
        "function": {
            "name": "list_files_info_in_dir",
            "description": "this function lists information of files in a directory",
            "parameters": {
                "type": "OBJECT",
                "properties": {
                    "directory": {"type": "STRING", "description": "an absolute path of a directory"}
                },
                "required": ["directory"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "count_lines_of_file",
            "description": "this function counts the number of lines in a file", 
            "parameters": {
                "type": "OBJECT",
                "properties": {
                    "directory": {"type": "STRING", "description": "an absolute path of a directory"},
                    "filename": {"type": "STRING"}
                },
                "required": ["directory", "filename"]
            }
        }
    }]' \
    --tool-callbacks="list_files_info_in_dir:/path/to/list_files_info_in_dir.sh" \
    --tool-callbacks="count_lines_of_file:/path/to/count_lines_of_file.sh" \
    --recurse-on-callback-results
```

Note that the mode of function calling config here is set to `AUTO`. If it is `ANY`, it may loop infinitely on the same function call result.

You can omit `--recurse-on-callback-results` / `-r` if you don't need it, but then it will just print the first function call result and exit.

#### Generate with Predefined Callbacks

You can set predefined callbacks for tool callbacks instead of scripts/binaries.

Here are predefined callback names:

* `@stdin`: Ask the user for standard input.
* `@format`: Print a formatted string with the resulting function arguments.
* … (more to be added)

##### @stdin

```bash
$ oll -m "mistral-small3.2:24b" \
    -p "send an email to steve that i'm still alive (ask me if you don't know steve's email address)" \
    --tools='[
        {
            "type": "function",
            "function": {
                "name": "send_email",
                "description": "this function sends an email with given values",
                "parameters": {
                    "type": "OBJECT",
                    "properties": {
                        "email_address": {"type": "STRING", "description": "email address of the recipient"},
                        "email_title": {"type": "STRING", "description": "email title"},
                        "email_body": {"type": "STRING", "description": "email body"}
                    },
                    "required": ["email_address", "email_title", "email_body"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "ask_email_address",
                "description": "this function asks for the email address of recipient"
            }
        }
    ]' \
    --tool-callbacks="send_email:/path/to/send_email.sh" \
    --tool-callbacks="ask_email_address:@stdin" \
    --recurse-on-callback-results
```

##### @format

With `--tool-callbacks="YOUR_CALLBACK:@format=YOUR_FORMAT_STRING"`, it will print the resulting function arguments as a string formatted with the [text/template](https://pkg.go.dev/text/template) syntax:

```bash
$ oll -m "mistral-small3.2:24b" \
    -p "categorize this image" \
    -f /some/image/file.jpg \
    --tools='[{
        "type": "function",
        "function": {
            "name": "categorize_image",
            "description": "this function categorizes the provided image",
            "parameters": {
                "type": "OBJECT",
                "properties": {
                    "category": {
                        "type": "STRING",
                        "description": "the category of the provided image",
                        "enum": ["animal", "person", "scenary", "object", "other"],
                        "nullable": false
                    },
                    "description": {
                        "type": "STRING",
                        "description": "the detailed description of the provided image",
                        "nullable": false
                    }
                },
                "required": ["category", "description"]
            }
        }}]' \
    --tool-callbacks='categorize_image:@format={{printf "Category: %s\nDescription: %s\n" .category .description}}' \
    --show-callback-results 2>/dev/null
```

When the format string is omitted (`--tool-callbacks="YOUR_CALLBACK:@format"`), it will be printed as a JSON string.

### Function Call with MCP

### Streamable HTTP URLs

```bash
$ oll -m "mistral-small3.2:24b" \
    -p "what is shoebill? search from the web" \
    --mcp-streamable-url="https://server.smithery.ai/exa/mcp?api_key=xxxxx&profile=yyyyy" \
    --recurse-on-callback-results
```

You can use `--mcp-streamable-url` multiple times for using multiple servers' functions:

```bash
$ oll -m "mistral-small3.2:24b" \
    -p '1. get any one github repository of @meinside
2. search for the respository name from web
3. then summarize the searched results' \
    --mcp-streamable-url="https://server.smithery.ai/exa/mcp?api_key=xxxxx&profile=yyyyy" \
    --mcp-streamable-url="https://server.smithery.ai/@smithery-ai/github/mcp?api_key=xxxxx&profile=yyyyy" \
    --recurse-on-callback-results
```

You can even mix tools from local and MCP servers:

```bash
$ oll -m "mistral-small3.2:24b" \
    -p "summarize the latest commits of repository 'oll' of github user @meinside (branch: master) and send them as an email to asdf@zxcv.net" \
    --mcp-streamable-url="https://server.smithery.ai/@smithery-ai/github/mcp?api_key=xxxxx&profile=yyyyy" \
    --tools='[{
        "type": "function",
        "function":{
            "name": "send_email",
            "description": "this function sends an email with given values",
            "parameters": {
                "type": "OBJECT",
                "properties": {
                    "email_address": {"type": "STRING", "description": "email address of the recipient"},
                    "email_title": {"type": "STRING", "description": "email title"},
                    "email_body": {"type": "STRING", "description": "email body"}
                },
                "required": ["email_address", "email_title", "email_body"]
            }
        }
    }]' \
    --tool-callbacks="send_email:/path/to/send_email.sh" \
    --recurse-on-callback-results
```

### Fetch URL Contents from the Prompt

Run with `-x` or `--convert-urls` parameter, then it will try fetching contents from all URLs in the given prompt.

Supported content types are:

* `text/*` (eg. `text/html`, `text/csv`, …)
* `application/json`

```bash
# generate with a text prompt which includes some urls in it 
$ oll -x \
    -p "what's the latest book of douglas adams? check from here: https://openlibrary.org/search/authors.json?q=douglas%20adams"
# NOTE: there might be a warning: "truncating input prompt"
```

### Generation with Multimodal Models

You can use [vision models](https://ollama.com/search?c=vision) with `-m` or `--model` parameter.

```bash
# generate with a multimodal model and image file(s)
$ oll -m gemma3:12b \
    -p "what is this picture?" \
    -f ~/Downloads/some_image.png
```

### Generating Embeddings

You can print [embeddings](https://ollama.com/search?c=embedding) of a given prompt in JSON format.

```bash
# generate with an embedding model
$ oll -e \
    -m nomic-embed-text \
    -p "this is an apple"
```

### Others

With verbose flags (`-v`, `-vv`, and `-vvv`) you can see more detailed information like generation metrics and request parameters.

## Known Issues

### Prompt Gets Truncated Unexpectedly

When the input prompt exceeds the context window, it [gets truncated silently](https://github.com/ollama/ollama/issues/7043) with warnings like this:

```bash
level=WARN source=runner.go:129 msg="truncating input prompt" limit=2048 prompt=2565 keep=5 new=2048
```

Try again with `-w` or `--context-window-size` parameter, or trim the prompt manually before generation.

## License

MIT

