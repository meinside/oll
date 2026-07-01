// serve_test.go

package main

import (
	"context"
	"testing"
	"time"
)

// test `runShellCommandWithContext` with shell features (pipes, redirections,
// logical operators, variable expansion, etc.)
func TestRunShellCommandWithContext(t *testing.T) {
	type test struct {
		cmdline  string
		stdout   string
		exitCode int
	}

	tests := []test{
		// plain command + args
		{
			cmdline:  `echo hello world`,
			stdout:   "hello world\n",
			exitCode: 0,
		},
		// pipe
		{
			cmdline:  `printf 'foo\nbar\nbaz\n' | grep ba | wc -l | tr -d ' '`,
			stdout:   "2\n",
			exitCode: 0,
		},
		// logical operators
		{
			cmdline:  `true && echo ok`,
			stdout:   "ok\n",
			exitCode: 0,
		},
		{
			cmdline:  `false || echo fallback`,
			stdout:   "fallback\n",
			exitCode: 0,
		},
		// variable expansion
		{
			cmdline:  `X=42; echo "value=$X"`,
			stdout:   "value=42\n",
			exitCode: 0,
		},
		// command substitution
		{
			cmdline:  `echo "count=$(printf 'a\nb\nc\n' | wc -l | tr -d ' ')"`,
			stdout:   "count=3\n",
			exitCode: 0,
		},
		// non-zero exit code is propagated
		{
			cmdline:  `exit 3`,
			stdout:   "",
			exitCode: 3,
		},
	}

	ctx := context.Background()
	for _, test := range tests {
		stdout, _, exitCode, _ := runShellCommandWithContext(ctx, test.cmdline)

		if stdout != test.stdout {
			t.Errorf("commandline '%s': expected stdout %q, got %q", test.cmdline, test.stdout, stdout)
		}
		if exitCode != test.exitCode {
			t.Errorf("commandline '%s': expected exit code %d, got %d", test.cmdline, test.exitCode, exitCode)
		}
	}
}

// test that `runShellCommandWithContext` honors context cancellation (timeout)
func TestRunShellCommandWithContextTimeout(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
	defer cancel()

	_, _, exitCode, err := runShellCommandWithContext(ctx, `sleep 5`)

	if err == nil {
		t.Errorf("expected an error due to timeout, got nil (exit code %d)", exitCode)
	}
	if ctx.Err() != context.DeadlineExceeded {
		t.Errorf("expected context deadline exceeded, got %v", ctx.Err())
	}
}
