# Speedmap Workflow

This repository contains an Ecoscope workflow for Speedmap, including 
- Download subject trajectories from Earth Ranger
- Calculate and classify subject moving speed
- Generate a speedmap dashboard with speed labels


## Run the Workflow

1. **Open the ecoscope-workflow-console App**

2. **Load the [Speedmap Workflow](https://github.com/ecoscope-platform-workflows-releases/speedmap)**

3. **Update the configuration**   
   Update the following parameters according to your requirements
   - time_range
   - er_client_name: data_source
   - subject_obs: subject_group_name
   - base_map_defs

4. **Set Mock IO**   
   By default, the workflow will use Mock IO. Leave it on if you don't have EarthRanger credentials and ONLY want to test the workflow functionality. Note that if you leave the Mock IO on, the above configuration will NOT take effect.

   For production use with real data, click on the button to disable "Mock IO"

5. **Run the workflow and wait for the dashboard to pop up**


## Test Steps with Pixi

1. **Install pixi**
   
   If you don't have pixi installed, install it by following the instructions at the [official documentation](https://pixi.sh/latest/).

2. **Test your workflow**
   
   Run the following command to test your workflow:
   ```bash
   pixi run pytest-cli
   ```