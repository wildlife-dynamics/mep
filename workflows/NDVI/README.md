# NDVI Workflow

This repository contains an Ecoscope workflow for Normalized Difference Vegetation Index (NDVI), including 
- Download Region of Interest(ROI) from a URL
- Calculate NDVI with Google Earth Engine
- Generate a dashboard with historical comparison


## Run the Workflow

1. **Open the ecoscope-workflow-console App**

2. **Load the [NDVI Workflow](https://github.com/ecoscope-platform-workflows-releases/ndvi)**

3. **Update the configuration**   
   Update the following parameters according to your requirements
   - time_range
   - roi:url Update your ROI package to dropbox, set it to be public and put the url here

4. **Set Mock IO**   
   By default, the workflow will use Mock IO. Leave it on if you don't have Google Earth Engine credentials and want to test the workflow functionality. Note that if you leave the Mock IO on, the above configuration will NOT take effect.

   For production use with real data, click on the button to disable "Mock IO"

5. **Run the workflow and wait for the dashboard to pop up**


## Set up Google Earth Engine credentials

1. **Set up Google Earth Engine Project and Authenticate**
To run the workflow, you need a google earth engine credential with a project id

   a. Sign up for Google Earth Engine at [https://earthengine.google.com/signup/](https://earthengine.google.com/signup/)
   b. Install the Earth Engine Python API:
      ```bash
      pip install earthengine-api
      ```
   c. Authenticate with your browser
      ```bash
      earthengine authenticate
      ```
   d. Initialize the Earth Engine API:
      ```bash
      earthengine init
      ```

2. **Set up environment**
   ```bash
   # Google EarthEngine Configuration
   export ecoscope_workflows__connections__ecoscope_poc__ee_project=<YOUR_GEE_PROJECT_ID>
   ```

## Test Steps

1. **Install pixi**
   
   If you don't have pixi installed, install it by following the instructions at the [official documentation](https://pixi.sh/latest/).

2. **Test your workflow**
   
   Run the following command to test your workflow:
   ```bash
   pixi run pytest-cli
   ```