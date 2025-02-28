# SIXSENSE mini

**Prescriptive AI for Actionable Recommendations**

SIXSENSE mini is a prescriptive AI application that goes beyond predictions to provide actionable recommendations. It helps users not only forecast outcomes but also prescribe optimal actions to improve decision-making.

![Image](https://github.com/user-attachments/assets/88b650ed-997f-4417-ad7d-7ef5d7a8926e)

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Routing Structure](#routing-structure)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Docker Deployment](#docker-deployment)
- [Usage Instructions](#usage-instructions)
- [License](#license)

## Overview

SIXSENSE mini leverages advanced prescriptive analytics to turn data into actionable strategies. It supports end-to-end processes from dataset upload and property configuration to model training and optimization result analysis.

## Features

- **Projects:** Create and manage projects.
- **Project (Flow):** Build and configure flows within a project.
- **Select Datasets:** Upload datasets, select those for analysis, and change property types via drag-and-drop.
- **Analyze Properties:** View histograms and verify flow metadata for each property.
- **Configure Properties:** Arrange properties according to your flow optimization scenario.
  - **Environmental:** Non-adjustable environment variables.
  - **Controllable:** Variables that can be adjusted.
  - **Output:** Standard output values within a dataset.
- **Set Goals:** Configure optimization objective functions and set range limits.
- **Set Priorities:** Assign priorities for each optimization objective.
- **Check Performance:** Review surrogate model performance metrics and inspect prediction cases.
- **Optimization Results:** View sample outcomes from the optimization process.

## Routing Structure

The application routing is organized as follows:

- **Datasets**
  - `/datasets/select` – Select Datasets Page
- **Properties**
  - `/properties/analyze` – Analyze Properties Page
  - `/properties/configure` – Configure Properties Page
- **Models**
  - `/models/set-goals` – Set Goals Page
  - `/models/set-priorities` – Set Priorities Page
  - `/models/training-progress` – Model Training Progress Page
- **Results**
  - `/results/check-performance` – Check Performance Page
  - `/results/optimization-results` – Optimization Results Page

## Project Structure

```bash
user-interface/
├── public/                # Contains static assets and the main index.html.
├── src/
│   ├── assets/            # Images, fonts, and other static resources.
│   ├── components/        # Reusable UI components (e.g., Card, Separator).
│   ├── layouts/           # Layout components that define the page structure.
│   ├── store/             # Redux store configuration, actions, and reducers.
│   ├── theme/             # Custom theme settings for Chakra UI.
│   ├── view/              # Page components corresponding to application routes.
│   ├── index.js           # The application’s entry point.
│   └── routes.js          # Route definitions and configuration.
├── .env                   # Environment-specific configuration variables.
└── package.json           # Project dependencies and build scripts.

```

## Installation

1. **Clone the Repository:**

   Clone the repository to your local machine and navigate to the project folder:

   ```bash
   git clone https://github.com/boostcampaitech7/level4-cv-finalproject-hackathon-cv-06-lv3.git
   cd user-interface/
   ```

2. **Install Dependencies:**

   Ensure that Node.js is installed on your system. Then, install the project dependencies by running:

   ```bash
   npm install
   ```

   Alternatively, if you prefer using Yarn:

   ```bash
   yarn install
   ```

3. **Environment Configuration:**

   Create or update the `.env` file in the project root with your environment-specific settings. For example, to set the API base URL, add the following line:

   ```env
   REACT_APP_API_BASE_URL=https://your-api-server.com:port
   ```

4. **Run the Application:**

   Start the development server by executing:

   ```
   npm start
   ```

   The application will launch in development mode. Open http://localhost:3000 in your browser to view the app.

## Docker Deployment

This project includes Docker support for containerizing the application. You can build a Docker image and run the application in a container using Docker and Docker Compose.

### Dockerfile Overview

The provided `Dockerfile` uses a multi-stage build process:

1. **Build Stage:**
   - Uses a Node.js image to install dependencies and build the React application.
   - The build output is generated in the `build` folder.
2. **Production Stage:**

   - Uses an Nginx image to serve the static files from the build folder.
   - The final image is lightweight and optimized for production.

3. **Environment Variables:**
   - Environment variables can be set via the `.env` file or directly in the Dockerfile using the `ENV` command.

### docker-compose.yml Overview

The `docker-compose.yml` file simplifies running the application container. It also leverages an `.env` file to pass environment-specific variables to the container.

Example `docker-compose.yml`:

```yaml
services:
  react:
    build: .
    volumes:
      - react-build:/app/build
    environment:
      - REACT_APP_API_BASE_URL = "http://localhost:8000"

  nginx:
    image: nginx:stable-alpine
    ports:
      - "80:80"
    volumes:
      - react-build:/usr/share/nginx/html:ro
    depends_on:
      - react

volumes:
  react-build:
```

### How to Run the Application with Docker

1. **Build the Docker Image:**

   In the project root (where your `Dockerfile` and `docker-compose.yml` are located), run:

   ```bash
   docker compose build
   ```

2. **Start the Docker Container:**

   Launch the container in detached mode with:

   ```bash
   docker compose up -d
   ```

3. **Access the Application:**

   Open your browser and navigate to http://localhost (or use your host's IP address) to view the application.

4. **Stop and Remove the Container:**

   When you're done, stop and remove the container by running:

   ```bash
   docker compose down
   ```

## Usage Instructions

The application is divided into several key pages, each corresponding to a specific stage of your workflow:

- **Projects:**  
  Create and manage your projects.

- **Project (Flow):**  
  Within a project, create a new flow.

- **Select Datasets:**  
  Upload datasets, select the ones to analyze, and modify property types using the drag-and-drop interface.

- **Analyze Properties:**  
  Review histograms for each property and verify flow metadata.

- **Configure Properties:**  
  Arrange properties according to your optimization scenario.

  - _Environmental:_ Non-adjustable variables
  - _Controllable:_ Variables that can be adjusted
  - _Output:_ Standard output values within a dataset

- **Set Goals:**  
  Configure the optimization objective functions and set limits on the optimization range.

  - For **Controllable** properties, you can choose **Minimize**, **Maximize**, or **Fit to Range**.
  - For **Output** properties, only **Fit to Property** is available.
  - _Additional goal-setting features are under development._

- **Set Priorities:**  
  Assign priorities for each optimization objective.

- **Check Performance:**  
  Evaluate surrogate model performance metrics and inspect prediction cases.

- **Optimization Results:**  
  View sample outcomes from the optimization process.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE.md) file for details.
