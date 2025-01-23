// helpers.go
//
// helper functions and constants

package main

import (
	"bytes"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"regexp"
	"slices"
	"strings"
	"time"

	"github.com/PuerkitoBio/goquery"
	"github.com/gabriel-vasile/mimetype"
	"github.com/ollama/ollama/api"
	"github.com/tailscale/hujson"
)

const (
	// for replacing URLs in prompt to body texts
	urlRegexp       = `https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)`
	urlToTextFormat = "<link url=\"%[1]s\" content-type=\"%[2]s\">\n%[3]s\n</link>"
)

// file/directory names to ignore while recursing directories
var _namesToIgnore = []string{
	"/", // NOTE: ignore root
	".cache/",
	".config/",
	".DS_Store",
	".env",
	".env.local",
	".git/",
	".ssh/",
	".svn/",
	".Trash/",
	".venv/",
	"build/",
	"config.json", "config.toml", "config.yaml", "config.yml",
	"Thumbs.db",
	"dist/",
	"node_modules/",
	"target/",
	"tmp/",
}
var _fileNamesToIgnore, _dirNamesToIgnore map[string]bool

// initialize things
func init() {
	// files and directories' names to ignore
	_fileNamesToIgnore, _dirNamesToIgnore = map[string]bool{}, map[string]bool{}
	for _, name := range _namesToIgnore {
		if strings.HasSuffix(name, "/") {
			_dirNamesToIgnore[filepath.Dir(name)] = true
		} else {
			_fileNamesToIgnore[name] = true
		}
	}
}

// standardize given JSON (JWCC) bytes
func standardizeJSON(b []byte) ([]byte, error) {
	ast, err := hujson.Parse(b)
	if err != nil {
		return b, err
	}
	ast.Standardize()

	return ast.Pack(), nil
}

// check if given directory should be ignored
func ignoredDirectory(path string) bool {
	if _, exists := _dirNamesToIgnore[filepath.Base(path)]; exists {
		logMessage(verboseMedium, "Ignoring directory '%s'", path)
		return true
	}
	return false
}

// check if given file should be ignored
func ignoredFile(path string, stat os.FileInfo) bool {
	// ignore empty files,
	if stat.Size() <= 0 {
		logMessage(verboseMedium, "Ignoring empty file '%s'", path)
		return true
	}

	// ignore files with ignored names,
	if _, exists := _fileNamesToIgnore[filepath.Base(path)]; exists {
		logMessage(verboseMedium, "Ignoring file '%s'", path)
		return true
	}

	return false
}

// return all files' paths in the given directory
func filesInDir(dir string, vbs []bool) ([]*string, error) {
	var files []*string

	// traverse directory
	err := filepath.WalkDir(dir, func(path string, d os.DirEntry, err error) error {
		if d.IsDir() {
			if ignoredDirectory(path) {
				return filepath.SkipDir
			}
		} else {
			stat, err := os.Stat(path)
			if err != nil {
				return err
			}

			if ignoredFile(path, stat) {
				return nil
			}

			logVerbose(verboseMedium, vbs, "attaching file '%s'", path)

			files = append(files, &path)
		}

		return nil
	})

	return files, err
}

// expand given filepaths (expand directories with their sub files)
func expandFilepaths(p params) (expanded []*string, err error) {
	filepaths := p.Filepaths
	if filepaths == nil {
		return nil, nil
	}

	// expand directories with their sub files
	expanded = []*string{}
	for _, fp := range filepaths {
		if fp == nil {
			continue
		}

		if stat, err := os.Stat(*fp); err == nil {
			if stat.IsDir() {
				if files, err := filesInDir(*fp, p.Verbose); err == nil {
					expanded = append(expanded, files...)
				} else {
					return nil, fmt.Errorf("failed to list files in '%s': %w", *fp, err)
				}
			} else {
				if ignoredFile(*fp, stat) {
					continue
				}
				expanded = append(expanded, fp)
			}
		} else {
			return nil, err
		}
	}

	// filter filepaths by supported mime types
	filtered := []*string{}
	for _, fp := range expanded {
		if fp == nil {
			continue
		}

		if matched, supported, err := supportedMimeTypePath(*fp); err == nil {
			if supported {
				filtered = append(filtered, fp)
			} else {
				logMessage(verboseMedium, "Ignoring file '%s', unsupported mime type: %s", *fp, matched)
			}
		} else {
			return nil, fmt.Errorf("failed to check mime type of '%s': %w", *fp, err)
		}
	}

	// remove redundant paths
	filtered = uniqPtrs(filtered)

	logVerbose(verboseMedium, p.Verbose, "attaching %d unique file(s)", len(filtered))

	return filtered, nil
}

// replace all HTTP URLs in `p.Prompt` to the content of each URL.
//
// files that were not converted to text will be returned as `files`.
func replaceURLsInPrompt(conf config, p params) (replaced string, files map[string][]byte) {
	userAgent := *p.UserAgent
	prompt := *p.Prompt
	vbs := p.Verbose

	files = map[string][]byte{}

	re := regexp.MustCompile(urlRegexp)
	for _, url := range re.FindAllString(prompt, -1) {
		if fetched, contentType, err := fetchContent(conf.ReplaceHTTPURLTimeoutSeconds, userAgent, url, vbs); err == nil {
			if supportedTextContentType(contentType) { // if it is a text of supported types,
				logVerbose(verboseMaximum, vbs, "text content (%s) fetched from '%s' is supported", contentType, url)

				// replace prompt text
				prompt = strings.Replace(prompt, url, fmt.Sprintf("%s\n", string(fetched)), 1)
			} else if mimeType, supported, _ := supportedMimeType(fetched); supported { // if it is a file of supported types,
				logVerbose(verboseMaximum, vbs, "file content (%s) fetched from '%s' is supported", mimeType, url)

				// replace prompt text,
				prompt = strings.Replace(prompt, url, fmt.Sprintf(urlToTextFormat, url, mimeType, ""), 1)

				// and add bytes as a file
				files[url] = fetched
			} else { // otherwise, (not supported in anyways)
				logVerbose(verboseMaximum, vbs, "fetched content (%s) from '%s' is not supported", contentType, url)
			}
		} else {
			logVerbose(verboseMedium, vbs, "failed to fetch content from '%s': %s", url, err)
		}
	}

	return prompt, files
}

// fetch the content from given url and convert it to text for prompting.
func fetchContent(timeoutSeconds int, userAgent, url string, vbs []bool) (converted []byte, contentType string, err error) {
	client := &http.Client{
		Timeout: time.Duration(timeoutSeconds) * time.Second,
	}

	logVerbose(verboseMaximum, vbs, "fetching content from '%s'", url)

	req, err := http.NewRequest("GET", url, nil)
	if err != nil {
		return nil, contentType, fmt.Errorf("failed to create http request: %w", err)
	}
	req.Header.Set("User-Agent", userAgent)

	resp, err := client.Do(req)
	if err != nil {
		return nil, contentType, fmt.Errorf("failed to fetch contents from '%s': %w", url, err)
	}
	defer func() {
		if err := resp.Body.Close(); err != nil {
			logError("Failed to close response body: %s", err)
		}
	}()

	// NOTE: get the content type from the header, not inferencing from the body bytes
	contentType = resp.Header.Get("Content-Type")

	logVerbose(verboseMaximum, vbs, "fetched content (%s) from '%s'", contentType, url)

	if resp.StatusCode == 200 {
		if supportedTextContentType(contentType) {
			if strings.HasPrefix(contentType, "text/html") {
				var doc *goquery.Document
				if doc, err = goquery.NewDocumentFromReader(resp.Body); err == nil {
					// NOTE: removing unwanted things here
					_ = doc.Find("script").Remove()                   // javascripts
					_ = doc.Find("link[rel=\"stylesheet\"]").Remove() // css links
					_ = doc.Find("style").Remove()                    // embeded css tyles

					converted = []byte(fmt.Sprintf(urlToTextFormat, url, contentType, removeConsecutiveEmptyLines(doc.Text())))
				} else {
					converted = []byte(fmt.Sprintf(urlToTextFormat, url, contentType, "Failed to read this HTML document."))
					err = fmt.Errorf("failed to read document (%s) from '%s': %w", contentType, url, err)
				}
			} else if strings.HasPrefix(contentType, "text/") {
				var bytes []byte
				if bytes, err = io.ReadAll(resp.Body); err == nil {
					converted = []byte(fmt.Sprintf(urlToTextFormat, url, contentType, removeConsecutiveEmptyLines(string(bytes)))) // NOTE: removing redundant empty lines
				} else {
					converted = []byte(fmt.Sprintf(urlToTextFormat, url, contentType, "Failed to read this document."))
					err = fmt.Errorf("failed to read document (%s) from '%s': %w", contentType, url, err)
				}
			} else if strings.HasPrefix(contentType, "application/json") {
				var bytes []byte
				if bytes, err = io.ReadAll(resp.Body); err == nil {
					converted = []byte(fmt.Sprintf(urlToTextFormat, url, contentType, string(bytes)))
				} else {
					converted = []byte(fmt.Sprintf(urlToTextFormat, url, contentType, "Failed to read this document."))
					err = fmt.Errorf("failed to read document (%s) from '%s': %w", contentType, url, err)
				}
			} else {
				converted = []byte(fmt.Sprintf(urlToTextFormat, url, contentType, fmt.Sprintf("Content type '%s' not supported.", contentType)))
				err = fmt.Errorf("content (%s) from '%s' not supported", contentType, url)
			}
		} else {
			if converted, err = io.ReadAll(resp.Body); err == nil {
				if matched, supported, _ := supportedMimeType(converted); !supported {
					converted = []byte(fmt.Sprintf(urlToTextFormat, url, matched, fmt.Sprintf("Content type '%s' not supported.", matched)))
					err = fmt.Errorf("content (%s) from '%s' not supported", matched, url)
				}
			} else {
				converted = []byte(fmt.Sprintf(urlToTextFormat, url, contentType, "Failed to read this file."))
				err = fmt.Errorf("failed to read file (%s) from '%s': %w", contentType, url, err)
			}
		}
	} else {
		converted = []byte(fmt.Sprintf(urlToTextFormat, url, contentType, fmt.Sprintf("HTTP Error %d", resp.StatusCode)))
		err = fmt.Errorf("http error %d from '%s'", resp.StatusCode, url)
	}

	logVerbose(verboseMaximum, vbs, "fetched body =\n%s", string(converted))

	return converted, contentType, err
}

// remove consecutive empty lines for compacting prompt lines
func removeConsecutiveEmptyLines(input string) string {
	// trim each line
	trimmed := []string{}
	for _, line := range strings.Split(input, "\n") {
		trimmed = append(trimmed, strings.TrimRight(line, " "))
	}
	input = strings.Join(trimmed, "\n")

	// remove redundant empty lines
	regex := regexp.MustCompile("\n{2,}")
	return regex.ReplaceAllString(input, "\n")
}

// check if given HTTP content type is a supported text type
func supportedTextContentType(contentType string) bool {
	return func(contentType string) bool {
		switch {
		case strings.HasPrefix(contentType, "text/"):
			return true
		case strings.HasPrefix(contentType, "application/json"):
			return true
		default:
			return false
		}
	}(contentType)
}

// get pointer of given value
func ptr[T any](v T) *T {
	val := v
	return &val
}

// get unique elements of given slice of pointers
func uniqPtrs[T comparable](slice []*T) []*T {
	keys := map[T]bool{}
	list := []*T{}
	for _, entry := range slice {
		if _, value := keys[*entry]; !value {
			keys[*entry] = true
			list = append(list, entry)
		}
	}
	return list
}

// convert given prompt & files for generation
func convertPromptAndFiles(prompt string, filesInPrompt map[string][]byte, filepaths []*string) (convertedPrompt string, images []api.ImageData, err error) {
	images = []api.ImageData{}
	files := map[string][]byte{}

	for url, file := range filesInPrompt {
		if isImage, _ := supportedImage(file); isImage {
			images = append(images, api.ImageData(file))
		} else {
			files[url] = file
		}
	}
	for _, fp := range filepaths {
		if opened, err := os.Open(*fp); err == nil {
			defer opened.Close()

			fbase := filepath.Base(*fp)
			if bytes, err := io.ReadAll(opened); err == nil {
				if isImage, _ := supportedImagePath(*fp); isImage {
					images = append(images, api.ImageData(bytes))
				} else {
					files[fbase] = bytes
				}
			} else {
				return "", nil, fmt.Errorf("failed to read file for prompt: %w", err)
			}
		} else {
			return "", nil, fmt.Errorf("failed to open file for prompt: %w", err)
		}
	}

	// build up `convertedPrompt` with `files`
	context := "<contexts>\n"
	for location, file := range files {
		context += fmt.Sprintf("<file location=\"%s\">\n%s\n</file>\n", location, string(file))
	}
	context += "</contexts>\n"
	return fmt.Sprintf("%s\n%s", context, prompt), images, nil
}

func supportedImage(data []byte) (supported bool, err error) {
	var mimeType *mimetype.MIME
	if mimeType, err = mimetype.DetectReader(bytes.NewReader(data)); err == nil {
		mimeTypeString := mimeType.String()

		return (mimeTypeString == "image/png" || mimeTypeString == "image/jpeg"), nil
	}

	return false, err
}

func supportedImagePath(filepath string) (supported bool, err error) {
	var f *os.File
	if f, err = os.Open(filepath); err == nil {
		defer f.Close()

		var mimeType *mimetype.MIME
		if mimeType, err = mimetype.DetectReader(f); err == nil {
			mimeTypeString := mimeType.String()

			return (mimeTypeString == "image/png" || mimeTypeString == "image/jpeg"), nil
		}
	}

	return false, err
}

// detect and return the matched mime type of given bytes data and whether it's supported or not.
func supportedMimeType(data []byte) (matchedMimeType string, supported bool, err error) {
	var mimeType *mimetype.MIME
	if mimeType, err = mimetype.DetectReader(bytes.NewReader(data)); err == nil {
		matchedMimeType, supported = checkMimeType(mimeType)

		return matchedMimeType, supported, nil
	}

	return http.DetectContentType(data), false, err
}

// detect and return the matched mime type of given path and whether it's supported or not.
func supportedMimeTypePath(filepath string) (matchedMimeType string, supported bool, err error) {
	var f *os.File
	if f, err = os.Open(filepath); err == nil {
		defer f.Close()

		var mimeType *mimetype.MIME
		if mimeType, err = mimetype.DetectReader(f); err == nil {
			matchedMimeType, supported = checkMimeType(mimeType)

			return matchedMimeType, supported, nil
		}
	}

	return "", false, err
}

// check if given file's mime type is matched and supported
func checkMimeType(mimeType *mimetype.MIME) (matched string, supported bool) {
	return func(mimeType *mimetype.MIME) (matchedMimeType string, supportedMimeType bool) {
		matchedMimeType = mimeType.String() // fallback

		switch {
		case slices.ContainsFunc([]string{
			// images (LLaVA 1.6)
			//
			// https://ai.google.dev/gemini-api/docs/vision?lang=go#technical-details-image
			"image/png",
			"image/jpeg",
			//"image/webp",
			//"image/heic",
			//"image/heif",

			// audios
			//
			// https://ai.google.dev/gemini-api/docs/audio?lang=go#supported-formats
			//"audio/wav",
			//"audio/mp3",
			//"audio/aiff",
			//"audio/aac",
			//"audio/ogg",
			//"audio/flac",

			// videos
			//
			// https://ai.google.dev/gemini-api/docs/vision?lang=go#technical-details-video
			//"video/mp4",
			//"video/mpeg",
			//"video/mov",
			//"video/avi",
			//"video/x-flv",
			//"video/mpg",
			//"video/webm",
			//"video/wmv",
			//"video/3gpp",

			// document formats
			//
			// https://ai.google.dev/gemini-api/docs/document-processing?lang=go#technical-details
			//"application/pdf",
			"application/x-javascript", "text/javascript",
			"application/x-python", "text/x-python",
			"text/plain",
			"text/html",
			"text/css",
			"text/md",
			"text/csv",
			"text/xml",
			//"text/rtf",
		}, func(element string) bool {
			if mimeType.Is(element) { // supported,
				matchedMimeType = element
				return true
			}
			return false // matched but not supported,
		}): // matched,
			return matchedMimeType, true
		default: // not matched, or not supported
			return matchedMimeType, false
		}
	}(mimeType)
}
