# Contributing

Thank you for your interest in contributing to stable-worldmodel! This library is a community-driven project, and we greatly appreciate contributions of all kinds.

If you encounter any issues or have suggestions, please open an issue on our [issue tracker](https://github.com/rbalestr-lab/stable-worldmodel/issues). This allows us to address problems and gather feedback from the community.

For those who want to contribute code or documentation, you can submit a pull request. Below, you will find details on how to prepare and submit your pull request effectively.

## PR Tutorial

The preferred workflow for contributing to stable-worldmodel is to fork the main repository on GitHub, clone, and develop on a branch. Steps:

1. **Fork the repository**: Fork the project repository by clicking on the 'Fork' button near the top right of the page. This creates a copy of the code under your GitHub user account. For more details on how to fork a repository see [this guide](https://docs.github.com/en/get-started/quickstart/fork-a-repo).

2. **Clone your fork** of the stable-worldmodel repo from your GitHub account to your local disk:

   ```bash
   git clone git@github.com:YourLogin/stable-worldmodel.git
   cd stable-worldmodel
   ```

3. **Install the package** in editable mode with the development dependencies, as well as the pre-commit hooks that will run on every commit:

   ```bash
   # Using uv (recommended)
   uv pip install -e ".[dev,doc]" && pre-commit install

   # Or using pip
   pip install -e ".[dev,doc]" && pre-commit install
   ```

4. **Create a feature branch** to hold your development changes:

   ```bash
   git checkout -b my-feature
   ```

   Always use a feature branch. It's good practice to never work on the `main` branch!

5. **Develop the feature** on your feature branch. Add changed files using `git add` and then commit the changes using `git commit`:

   ```bash
   git add modified_files
   git commit -m "Your commit message here"
   ```

6. **Push the changes** to your GitHub account:

   ```bash
   git push -u origin my-feature
   ```

7. Follow [these instructions](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request-from-a-fork) to create a pull request from your fork. Then, a project maintainer will review your changes.

## PR Checklist

When preparing a PR, please make sure to check the following points:

### 1. Tests

- **All automatic tests pass** on your local machine. Run tests using:

  ```bash
  pytest --cov=stable_worldmodel --cov-report=term-missing
  ```

- **Add tests for your changes**. We use [pytest](https://docs.pytest.org/en/stable/getting-started.html) for testing. All new features and bug fixes should include appropriate test coverage.

- **Test files** should be placed in the `stable_worldmodel/tests/` directory and follow the naming convention `test_*.py`.

### 2. Documentation

Each PR should include comprehensive documentation:

- **Docstrings**: All new functions, classes, and methods must include docstrings following the [Google style](https://www.sphinx-doc.org/en/master/usage/extensions/example_google.html).

- **Documentation pages**: Update or add documentation in the `docs/source/` directory. We use [Sphinx](https://www.sphinx-doc.org/en/master/usage/quickstart.html) with the [Awesome Sphinx Theme](https://sphinxawesome.xyz/).

- **Build documentation locally** to verify your changes:

  ```bash
  cd docs
  make clean
  make html
  ```

  The resulting HTML files will be in `docs/build/html/` and are viewable in a web browser.

### 3. PR Description

- **Link to issues**: If your pull request addresses an issue, please use the pull request title to describe the issue and mention the issue number in the pull request description (e.g., "Fixes #123").

- **Explicitly describe changes**: Your PR description should clearly explain:
  - **What** changed (list of modifications)
  - **Why** the change was necessary (problem being solved)
  - **How** it works (implementation approach if non-trivial)

### 4. PR Prefixes

When creating a pull request, use the appropriate prefix to indicate its status and type:

**Status prefix:**

- `[WIP]` (Work in Progress): Use this prefix for an incomplete contribution where further work is planned before seeking a full review. Consider including a task list in your PR description to outline planned work or track progress. Remove the `[WIP]` prefix once the PR is ready for review.

**Type prefix** (use when PR is ready):

- `[BugFix]`: For bug fixes and patches
- `[Feature]`: For new features or enhancements
- `[Environment]`: For adding or updating world environments
- `[Benchmark]`: For benchmark-related changes
- `[Doc]`: For documentation updates
- `[Test]`: For test additions or improvements

**Examples:**

- `[WIP][Feature] Add new MPPI solver implementation`
- `[BugFix] Fix reward calculation in PushT environment`
- `[Environment] Add Maze2D environment`
- `[Doc] Update tutorial for DINO-WM reproduction`

A `[WIP]` PR can serve multiple purposes:

1. Indicate that you are actively working on something to prevent duplicated efforts by others.
2. Request early feedback on functionality, design, or API.
3. Seek collaborators to assist with development.

## Code Style

We use `ruff` for code formatting and linting. The pre-commit hooks will automatically format your code, but you can also run it manually:

```bash
# Format code
ruff format .

# Lint code
ruff check . --fix
```

## Adding New Environments

If you're contributing a new world environment:

1. Create your environment class in `stable_worldmodel/envs/`
2. Register it in `stable_worldmodel/__init__.py` using the `register()` function
3. Add comprehensive docstrings explaining the environment
4. Include a `variation_space` to define controllable factors
5. Add tests in `stable_worldmodel/tests/`
6. Create documentation in `docs/source/world/` describing:
   - Environment description and goal
   - Action and observation spaces
   - Difficulty
   - Variations and factors of variation
   - Example usage

## Testing

All contributions should include appropriate tests:

- **Unit tests**: Test individual functions and classes in isolation
- **Integration tests**: Test how components work together
- **Environment tests**: Ensure environments follow Gymnasium API

Run specific test files:

```bash
pytest stable_worldmodel/tests/test_world.py
```

Run tests with coverage report:

```bash
pytest --cov=stable_worldmodel --cov-report=term-missing
```

## New Contributor Tips

A great way to start contributing to stable-worldmodel is to:

1. **Pick a good first issue**: Check the [good first issues](https://github.com/rbalestr-lab/stable-worldmodel/labels/good%20first%20issue) in the issue tracker
2. **Improve documentation**: Fix typos, clarify explanations, or add examples
3. **Add tests**: Increase test coverage for existing functionality
4. **Review PRs**: Help review other contributors' pull requests

## Questions?

If you have questions about contributing, feel free to:

- Open a [discussion](https://github.com/rbalestr-lab/stable-worldmodel/discussions)
- Join our [Discord server](https://discord.com/invite/adzpqWKM25)
- Ask in an issue or pull request

Thank you for contributing to stable-worldmodel! 🌎
