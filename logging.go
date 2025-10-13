// logging.go
//
// things for logging messages

package main

import (
	"encoding/json"
	"fmt"
	"os"
	"strings"

	"github.com/fatih/color"
	"github.com/jessevdk/go-flags"
	"github.com/jwalton/go-supportscolor"
)

// verbosity level constants
type verbosity uint

const (
	verboseNone    verbosity = iota
	verboseMinimum verbosity = iota
	verboseMedium  verbosity = iota
	verboseMaximum verbosity = iota
)

// verboseLevel checks level of verbosity.
func verboseLevel(
	verbosityFromParams []bool,
) verbosity {
	if len(verbosityFromParams) == 1 {
		return verboseMinimum
	} else if len(verbosityFromParams) == 2 {
		return verboseMedium
	} else if len(verbosityFromParams) >= 3 {
		return verboseMaximum
	}

	return verboseNone
}

// output writer for managing printings to stdout/stderr
type outputWriter struct {
	endsWithNewLine bool
}

// newOutputWriter generates a new output writer.
func newOutputWriter() *outputWriter {
	return &outputWriter{
		endsWithNewLine: true,
	}
}

// println force-adds a new line.
func (w *outputWriter) println() {
	fmt.Println()
	w.endsWithNewLine = true
}

// makeSureToEndWithNewLine makes sure stdout ends with a new line.
func (w *outputWriter) makeSureToEndWithNewLine() {
	if !w.endsWithNewLine {
		w.println()
	}
}

// printColored prints given string to stdout with color (if possible).
func (w *outputWriter) printColored(
	c color.Attribute,
	format string,
	a ...any,
) {
	formatted := fmt.Sprintf(format, a...)

	if supportscolor.Stdout().SupportsColor { // if color is supported,
		c := color.New(c)
		_, _ = c.Print(formatted)
	} else {
		fmt.Print(formatted)
	}

	w.endsWithNewLine = strings.HasSuffix(formatted, "\n")
}

// errorColored prints given string to stderr with color (if possible).
func (w *outputWriter) errorColored(
	c color.Attribute,
	format string,
	a ...any,
) {
	formatted := fmt.Sprintf(format, a...)

	if supportscolor.Stderr().SupportsColor { // if color is supported,
		c := color.New(c)
		_, _ = c.Fprint(os.Stderr, formatted)
	} else {
		fmt.Fprint(os.Stderr, formatted)
	}

	w.endsWithNewLine = strings.HasSuffix(formatted, "\n")
}

// print given string to stdout (will add a new line if there isn't).
func (w *outputWriter) print(
	level verbosity,
	format string,
	a ...any,
) {
	if !strings.HasSuffix(format, "\n") {
		format += "\n"
	}

	var c color.Attribute
	switch level {
	case verboseMinimum:
		c = color.FgGreen
	case verboseMedium, verboseMaximum:
		c = color.FgYellow
	default:
		c = color.FgWhite
	}

	w.printColored(
		c,
		format,
		a...,
	)
}

// print verbose message (will add a new line if there isn't).
//
// (only when the level of given `verbosityFromParams` is greater or equal to `targetLevel`)
func (w *outputWriter) verbose(
	targetLevel verbosity,
	verbosityFromParams []bool,
	format string,
	a ...any,
) {
	if vb := verboseLevel(verbosityFromParams); vb >= targetLevel {
		format = fmt.Sprintf(">>> %s", format)

		w.print(
			targetLevel,
			format,
			a...,
		)
	}
}

// errWithNewlineAppended prints given string to stderr and appends a new line if there isn't.
func (w *outputWriter) errWithNewlineAppended(
	c color.Attribute,
	format string,
	a ...any,
) {
	if !strings.HasSuffix(format, "\n") {
		format += "\n"
	}

	w.errorColored(
		c,
		format,
		a...,
	)
}

// warn prints given warning string to stderr (will add a new line if there isn't).
func (w *outputWriter) warn(
	format string,
	a ...any,
) {
	w.errWithNewlineAppended(color.FgMagenta, format, a...)
}

// error prints given error string to stderr (will add a new line if there isn't).
func (w *outputWriter) error(
	format string,
	a ...any,
) {
	w.errWithNewlineAppended(color.FgRed, format, a...)
}

// printHelpBeforeExit prints help message before os.Exit().
func (w *outputWriter) printHelpBeforeExit(
	code int,
	parser *flags.Parser,
) (exit int) {
	parser.WriteHelp(os.Stdout)

	return code
}

// printErrorBeforeExit prints error before os.Exit().
func (w *outputWriter) printErrorBeforeExit(
	code int,
	format string,
	a ...any,
) (exit int) {
	if code > 0 {
		w.error(format, a...)
	}

	return code
}

// prettify given thing in JSON format.
func prettify(
	v any,
	flatten ...bool,
) string {
	if len(flatten) > 0 && flatten[0] {
		if bytes, err := json.Marshal(v); err == nil {
			return string(bytes)
		}
	} else {
		if bytes, err := json.MarshalIndent(v, "", "  "); err == nil {
			return string(bytes)
		}
	}
	return fmt.Sprintf("%+v", v)
}
