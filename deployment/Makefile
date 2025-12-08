.PHONY: help build run stop logs clean setup

help: ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Available targets:'
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  %-15s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

setup: ## Run initial setup (create .env and secrets directory)
	@./deployment/setup.sh

build: ## Build Docker image
	docker-compose build

run: ## Start the application
	docker-compose up -d

stop: ## Stop the application
	docker-compose down

restart: ## Restart the application
	docker-compose restart

logs: ## View application logs
	docker-compose logs -f

logs-inference: ## View inference service logs only
	docker-compose exec trash-optimizer tail -f /var/log/supervisor/inference.out.log

logs-webapp: ## View webapp service logs only
	docker-compose exec trash-optimizer tail -f /var/log/supervisor/webapp.out.log

shell: ## Open a shell in the container
	docker-compose exec trash-optimizer /bin/bash

clean: ## Stop and remove containers, images, and volumes
	docker-compose down -v
	docker rmi trash-optimizer:latest || true

rebuild: stop build run ## Rebuild and restart the application

status: ## Show container status
	docker-compose ps
