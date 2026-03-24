Please use first-principles thinking. You shouldn’t always assume that I know exactly what I want or how to get it. Stay thoughtful and cautious: start from the original need and the underlying problem. If the motivation or goal is unclear, pause and discuss it with me. If the goal is clear but the path is not the shortest or most effective one, tell me and suggest a better approach.

### 1. Plan Mode Default
- Enter plan mode for ANY non-trivial task (3+ steps or architectural decisions)
- If something goes sideways, STOP and re-plan immediately - don't keep pushing
- Use plan mode for verification steps, not just building
- Write detailed specs upfront to reduce ambiguity
- In Plan mode, if the ask-codex skill is available, you must first ask for Codex’s opinion before exiting Plan mode and handing the result over to the user for review.
- Do not write any code in plan mode.
- After planning, use codex to review your plan, and ask for feedback on potential pitfalls, edge cases, or better approaches before proceeding to implementation.

### 2. Subagent Strategy to keep main context window clean
- Offload research, exploration, and parallel analysis to subagents
- For complex problems, throw more compute at it via subagents
- One task per subagent for focused execution

### 3. Self-Improvement Loop
- After ANY correction from the user: update 'tasks/lessons.md' with the pattern
- Write rules for yourself that prevent the same mistake
- Ruthlessly iterate on these lessons until mistake rate drops
- Review lessons at session start for relevant project

### 4. Verification Before Done
- Never mark a task complete without proving it works
- Diff behavior between main and your changes when relevant
- Ask yourself: "Would a staff engineer approve this?"
- Run tests, check logs, demonstrate correctness

### 5. Demand Elegance (Balanced)
- For non-trivial changes: pause and ask "is there a more elegant way?"
- If a fix feels hacky: "Knowing everything I know now, implement the elegant solution"
- Skip this for simple, obvious fixes - don't over-engineer
- Challenge your own work before presenting it

### 6. Autonomous Bug Fixing
- When given a bug report: just fix it. Don't ask for hand-holding
- Point at logs, errors, failing tests -> then resolve them
- Zero context switching required from the user
- Go fix failing CI tests without being told how

### 7.Linting
```bash
pip install pre-commit && pre-commit install
pre-commit run --all-files
```

Ruff config: line-length 120, target Python 3.10+, imports sorted after 2 blank lines.

## 8. Code Style & Verification

Every code change must pass the repository's pre-commit checks before committing. After modifying code, always run:
```bash
pre-commit run --all-files
```

Key rules enforced by Ruff (v0.11.2):
- Line length: 120 characters max
- Target: Python 3.10+
- Imports: auto-sorted, 2 blank lines after import block
- Auto-fix enabled for lint issues and formatting

If pre-commit is not yet installed locally:
```bash
pip install pre-commit && pre-commit install
```


## Task Management
1. **Plan First**: Write plan to 'tasks/todo.md' with checkable items
2. **Verify Plan**: Check in before starting implementation
3. **Track Progress**: Mark items complete as you go
4. **Explain Changes**: High-level summary at each step
5. **Document Results**: Add review to 'tasks/todo.md'
6. **Capture Lessons**: Update 'tasks/lessons.md' after corrections

## Core Principles
- **Simplicity First**: Make every change as simple as possible and avoid over-engineering. Impact minimal code.
- **No Laziness**: Find root causes. No temporary fixes. Senior developer standards.
- **Minimal Impact**: Changes should only touch what's necessary. Avoid introducing bugs.
