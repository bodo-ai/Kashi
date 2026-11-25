# Repository Architecture Documentation

Every agent working in this repository must help maintain
the repository architecture document:

`PROJECT_MAP.md`

## Requirements

1. At the beginning of a task, consult PROJECT_MAP.md if it exists.

2. If PROJECT_MAP.md does not exist, create it by inspecting
   the repository.

3. Whenever you add, remove, rename, or substantially modify
   source files or directories, update PROJECT_MAP.md in
   the same task.

4. Document every source file and explain its responsibility.
   Include important relationships between components.

5. Keep the directory tree synchronized with the actual
   repository structure.

6. Do not simply list filenames. Explain what each file
   implements and why it exists.

7. When uncertain about a file's responsibility, inspect
   its contents and relevant callers rather than guessing.

8. Keep descriptions concise but technically informative.

9. Do not overwrite useful existing documentation.
   Make incremental updates.

10. Before completing a task, verify that PROJECT_MAP.md
    accurately reflects all structural changes made.
