// config.go
//
// things for configurations

package main

import (
	"encoding/json"
	"os"
	"path/filepath"
)

// config struct
type config struct {
	DefaultModel      *string `json:"default_model,omitempty"`
	SystemInstruction *string `json:"system_instruction,omitempty"`

	TimeoutSeconds int `json:"timeout_seconds,omitempty"`

	ReplaceHTTPURLTimeoutSeconds int `json:"replace_http_url_timeout_seconds,omitempty"`

	SmitheryAPIKey *string `json:"smithery_api_key,omitempty"`
}

// read config from given filepath
func readConfig(
	configFilepath string,
) (conf config, err error) {
	var bytes []byte

	bytes, err = os.ReadFile(configFilepath)
	if err == nil {
		bytes, err = standardizeJSON(bytes)
		if err == nil {
			err = json.Unmarshal(bytes, &conf)
			if err == nil {
				// set default values
				if conf.TimeoutSeconds <= 0 {
					conf.TimeoutSeconds = defaultTimeoutSeconds
				}
				if conf.ReplaceHTTPURLTimeoutSeconds <= 0 {
					conf.ReplaceHTTPURLTimeoutSeconds = defaultFetchURLTimeoutSeconds
				}
			}
		}
	}

	return conf, err
}

// resolve config filepath
func resolveConfigFilepath(
	configFilepath *string,
) string {
	if configFilepath != nil {
		return *configFilepath
	}

	configHome := os.Getenv("XDG_CONFIG_HOME")
	if configHome != "" {
		return filepath.Join(configHome, appName, defaultConfigFilename)
	}

	return filepath.Join(os.Getenv("HOME"), ".config", appName, defaultConfigFilename)
}
