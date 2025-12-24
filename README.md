# KAIA Demo

## Prerequisites

Before you begin, ensure you have [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running on your system.

## Getting Started

Follow these steps to get the application running.

### 1. Clone the Repository

First, clone the project files to your local machine with command prompt or terminal:

```sh
git clone https://github.com/MahmutOzgurKizil/KAIA_demo
cd KAIA_demo
```
*If you do not have git installed, you can also download the ZIP file from the GitHub repository and extract it. Then, navigate into the project directory:*


```sh
cd path/to/KAIA_demo
```

### 2. Build and Run the Services

Use `docker compose` to build the containers and start all services in the background.

```sh
docker compose up -d --build
```

### 3. Pull Required AI Models

For the _first-time_ setup, you need to pull the language and embedding models that the application uses. Run the following commands from your terminal:

```sh
# Pull the main language model (Llama 3.2)
docker compose exec ollama ollama pull llama3.2

# Pull the text embedding model
docker compose exec ollama ollama pull nomic-embed-text
```

The application is now set up and ready to use.

## Usage

Once the services are running, you can access the web interface by navigating to:

[http://localhost:8000](http://localhost:8000)

You can log in as either a student or an instructor to see the different views.

## Managing the Application

Here are the common commands for managing the application services:

- **To Stop the Application**:
  Stops and removes the containers.
  ```sh
  docker compose down
  ```

- **To Start the Application**:
  Starts the existing containers without rebuilding.
  ```sh
  docker compose up -d
  ```

- **To Restart the Application**:
  Quickly restarts all services.
  ```sh
  docker compose restart
  ```

- **To Rebuild and Restart**:
  If you make changes to the `api` code or `Dockerfile`, you'll need to rebuild the image.
  ```sh
  docker compose up -d --build
  ```
