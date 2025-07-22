// mcp.go

package main

import (
	"context"
	"encoding/json"
	"fmt"
	"net"
	"net/http"
	"time"

	"github.com/meinside/version-go"
	"github.com/modelcontextprotocol/go-sdk/mcp"
	"github.com/ollama/ollama/api"
)

const (
	mcpClientName = `oll/mcp`
)

// a map of MCP connections and tools
type mcpConnectionsAndTools map[string]struct {
	connection *mcp.ClientSession
	tools      []*mcp.Tool
}

// get a matched server url and tool from given MCP tools and function name
func mcpToolFrom(
	connsAndTools mcpConnectionsAndTools,
	fnName string,
) (serverURL string, mc *mcp.ClientSession, tool mcp.Tool, exists bool) {
	for serverURL, tools := range connsAndTools {
		for _, tool := range tools.tools {
			if tool != nil && tool.Name == fnName {
				return serverURL, tools.connection, *tool, true
			}
		}
	}

	return "", nil, mcp.Tool{}, false
}

// fetch function declarations from MCP server
func fetchMCPTools(
	ctx context.Context,
	mc *mcp.ClientSession,
) (tools []*mcp.Tool, err error) {
	var listed *mcp.ListToolsResult
	if listed, err = mc.ListTools(ctx, &mcp.ListToolsParams{}); err == nil {
		return listed.Tools, nil
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

// fetch function result from MCP
func fetchToolCallResult(
	ctx context.Context,
	mc *mcp.ClientSession,
	fnName string, fnArgs map[string]any,
) (res *mcp.CallToolResult, err error) {
	if res, err = mc.CallTool(
		ctx,
		&mcp.CallToolParams{
			Name:      fnName,
			Arguments: fnArgs,
		},
	); err == nil {
		return res, nil
	}

	return
}

// connect to MCP server, start, initialize, and return the client
func mcpConnect(
	ctx context.Context,
	url string,
) (connection *mcp.ClientSession, err error) {
	streamable := mcp.NewStreamableClientTransport(
		url,
		&mcp.StreamableClientTransportOptions{
			HTTPClient: mcpHTTPClient(),
		},
	)

	client := mcp.NewClient(
		&mcp.Implementation{
			Name:    mcpClientName,
			Version: version.Build(version.OS | version.Architecture),
		},
		&mcp.ClientOptions{},
	)

	if connection, err = client.Connect(ctx, streamable); err == nil {
		return connection, nil
	}

	return nil, err
}

const (
	mcpDefaultTimeoutSeconds               = 120 // FIXME: ideally, should be 0 for keeping the connection
	mcpDefaultDialTimeoutSeconds           = 10
	mcpDefaultKeepAliveSeconds             = 60
	mcpDefaultIdleTimeoutSeconds           = 180
	mcpDefaultTLSHandshakeTimeoutSeconds   = 20
	mcpDefaultResponseHeaderTimeoutSeconds = 60
	mcpDefaultExpectContinueTimeoutSeconds = 15
)

// for reusing http client
var _mcpHTTPClient *http.Client

// helper function for generating a http client
func mcpHTTPClient() *http.Client {
	if _mcpHTTPClient == nil {
		_mcpHTTPClient = &http.Client{
			Timeout: defaultTimeoutSeconds * time.Second,
			Transport: &http.Transport{
				DialContext: (&net.Dialer{
					Timeout:   mcpDefaultDialTimeoutSeconds * time.Second,
					KeepAlive: mcpDefaultKeepAliveSeconds * time.Second,
				}).DialContext,
				IdleConnTimeout:       mcpDefaultIdleTimeoutSeconds * time.Second,
				TLSHandshakeTimeout:   mcpDefaultTLSHandshakeTimeoutSeconds * time.Second,
				ResponseHeaderTimeout: mcpDefaultResponseHeaderTimeoutSeconds * time.Second,
				ExpectContinueTimeout: mcpDefaultExpectContinueTimeoutSeconds * time.Second,
				DisableCompression:    true,
			},
		}
	}
	return _mcpHTTPClient
}
