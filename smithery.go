// smithery.go

package main

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/meinside/smithery-go"
	"github.com/modelcontextprotocol/go-sdk/mcp"
	"github.com/ollama/ollama/api"
)

// get a matched server name and tool from given smithery tools and function name
func smitheryToolFrom(
	smitheryTools map[string][]*mcp.Tool,
	fnName string,
) (serverName string, tool mcp.Tool, exists bool) {
	for serverName, tools := range smitheryTools {
		for _, tool := range tools {
			if tool != nil && tool.Name == fnName {
				return serverName, *tool, true
			}
		}
	}

	return "", mcp.Tool{}, false
}

// get a new smithery client
func newSmitheryClient(
	smitheryAPIKey string,
) *smithery.Client {
	return smithery.NewClient(smitheryAPIKey)
}

// fetch function declarations from smithery
func fetchSmitheryTools(
	ctx context.Context,
	client *smithery.Client,
	smitheryProfileID, smitheryQualifiedServerName string,
) (tools []*mcp.Tool, err error) {
	var conn *mcp.ClientSession
	if conn, err = client.ConnectWithProfileID(
		ctx,
		smitheryProfileID,
		smitheryQualifiedServerName,
	); err == nil {
		defer conn.Close()

		var listed *mcp.ListToolsResult
		if listed, err = conn.ListTools(ctx, &mcp.ListToolsParams{}); err == nil {
			return listed.Tools, nil
		}
	}
	return
}

// convert MCP tools to Ollama tools
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
		if marshalled, err := f.InputSchema.MarshalJSON(); err == nil {
			if err := json.Unmarshal(marshalled, &to[i].Function.Parameters); err != nil {
				return nil, fmt.Errorf("could not convert json to function parameters: %w", err)
			}
		} else {
			return nil, fmt.Errorf("could not convert input schema to json: %w", err)
		}
	}

	return to, nil
}

// fetch function result from smithery
func fetchSmitheryToolCallResult(
	ctx context.Context,
	client *smithery.Client,
	smitheryProfileID, smitheryQualifiedServerName string,
	fnName string, fnArgs map[string]any,
) (res *mcp.CallToolResult, err error) {
	var conn *mcp.ClientSession
	if conn, err = client.ConnectWithProfileID(
		ctx,
		smitheryProfileID,
		smitheryQualifiedServerName,
	); err == nil {
		defer conn.Close()

		if res, err = conn.CallTool(
			ctx,
			&mcp.CallToolParams{
				Name:      fnName,
				Arguments: fnArgs,
			},
		); err == nil {
			return res, nil
		}
	}
	return
}
