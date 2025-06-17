# Mara Elephant Project Workflow Repository

This repository contains workflows for Mara Elephant Project and custom tasks. Follow the setup steps below to get started.


## Project Structure

- `src/ecoscope-workflows-ext-mep`: Contains custom tasks
- `workflows`: Contains all the workflows that are set up to use custom tasks
- `pixi.toml`: Contains project configuration including dependencies
- `dev`: Contains scripts required for development
- `publish`: Contains scripts required to build and publish the task package

## Task Development
The tasks are defined under `src/ecoscope-workflows-ext-mep/ecoscope_workflows_ext_mep/tasks`. Here's an example of a task

```python
@task
def add_one_thousand(
    value: Annotated[float, Field(default=0, description="value to add")] = 0
) -> float:
    return value+1000
```
Annotate your function with `@task` and input parameters with `Annotated[float, Field(default=0, description="value to add")] ` to indicate the type and the description of the input. 

Note: make sure to include your tasks to `__init__.py` to register it as a ecoscope-workflow task.

You can also add other dependencies in `src/ecoscope-workflows-ext-mep/pyproject.toml` under [tool.pixi.dependencies] section

Unittests are under `src/ecoscope-workflows-ext-mep/tests` and can be tested it by
```bash
pixi run pytest-310
```
or pytest-311/pytest-312 depending on your desired python version.



## Build the Task Package

1. Update the build recipe under `publish/recipes/release` to include extra dependencies
2. Run
   ```bash
   pixi run build-release
   ```

   Now you can use these tasks in your workflow by including the local channel
   ```toml
   [project]
   channels = [
      "file:///tmp/ecoscope-workflows-custom/release/artifacts/",
      ...
   ]
   ...
   [feature.compile.dependencies]
   ecoscope-workflows-ext-mep = { version = "*", channel = "file:///tmp/ecoscope-workflows-custom/release/artifacts/" }
   ```

## Workflow Development
1. Update your workflow, including
- spec.yaml: a list of tasks and their relationships
- param.yaml: default configuration
- layout.json: update the default dashboard layout if your workflow generates a dashboard

2. Run the following command to compile your workflow:
   ```bash
   pixi run compile-mep-monthly-report
   ```
   
   This will generate a folder called `ecoscope-workflows-mep-monthly-report-workflow` with your compiled workflow.

   Later on if you update the workflow spec you can recompile it using:
   ```bash
   pixi run recompile-mep-monthly-report
   ```

3. Test your workflow. First set up your output directory by
   ```bash
   mkdir -p /tmp/workflows/mep-collar-voltage/output
   export ECOSCOPE_WORKFLOWS_RESULTS=file:///tmp/workflows/mep-collar-voltage/output
   ```
   Then run your workflow by
   ```bash
   cd workflows/mep-monthly-report/ecoscope-workflows-mep-monthly-report-workflow
   pixi run ecoscope-workflows-mep-monthly-report-workflow run --config-file ../param.yaml --execution-mode sequential --mock-io
   ```
   You can find the results in your output folder in `/tmp/workflows/mep-monthly-report/output/result.json`
   
## Publish Your Changes

1. Bump the git commit version by

   ```bash
   git tag v0.0.2
   git push origin --tags
   ```

2. Build your task package again
   ```bash
   pixi run build-release
   ```

3. Publish your task package
   ```bash
   ./publish/push.sh
   ```

4. Check if your package exists on prefix.dev

5. Update the version in pixi.toml
   ```toml
   [dependencies.ecoscope-workflows-ext-mep]
   channel = 'https://repo.prefix.dev/ecoscope-workflows-custom/'
   version = '0.0.2'
   ```

6. Update the version in spec.yaml
   ```toml
   - name: ecoscope-workflows-ext-mep
     version: '0.0.2'
     channel: https://repo.prefix.dev/ecoscope-workflows-custom/
   ```

7. Push your workflow changes to github.

## Troubleshoot

1. *Task not Registered*
   Clean up pixi caches by
   ```bash
   pixi clean cache
   rm -rf .pixi
   rm -rf pixi.lock
   pixi update
   ```
   And compile again


## Additional Resources

- [Ecoscope Core Library](https://github.com/wildlife-dynamics/ecoscope)
- [Pixi Documentation](https://pixi.sh/latest/)
