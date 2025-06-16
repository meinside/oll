// run.go

package main

import (
	"context"
	"fmt"
	"os"
	"time"

	"github.com/jessevdk/go-flags"
	"github.com/meinside/version-go"
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
		logMessage(verboseMedium, "No task was requested.")

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
			logVerbose(verboseMaximum, p.Verbose, "embeddings request params with prompt: %s\n\n", prettify(p))

			return doEmbeddingsGeneration(context.TODO(), conf, p)
		} else {
			logVerbose(verboseMaximum, p.Verbose, "generation request params with prompt: %s\n\n", prettify(p))

			return doGeneration(context.TODO(), conf, p)
		}
	} else if p.ListModels {
		return doListModels(context.TODO(), conf, p)
	} else { // otherwise,
		logVerbose(verboseMaximum, p.Verbose, "falling back with params: %s\n\n", prettify(p))

		logMessage(verboseMedium, "Parameter error: no task was requested or handled properly.")

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
