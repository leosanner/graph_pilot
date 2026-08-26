include .env
export

# yoyo maps the `postgresql://` scheme to psycopg2; this project ships psycopg3
# only, so migrations run through the `postgresql+psycopg://` scheme.
YOYO_DATABASE_URL := postgresql+psycopg://$(POSTGRES_USER):$(POSTGRES_PASSWORD)@$(POSTGRES_HOST):$(POSTGRES_PORT)/$(POSTGRES_DB)

.PHONY: db-up db-down db-reset db-logs db-shell migrate rollback

db-up:
	docker compose up -d postgres

db-down:
	docker compose down

# drops the volume: extensions are recreated by the init script on the next up
db-reset:
	docker compose down -v
	docker compose up -d postgres

db-logs:
	docker compose logs -f postgres

db-shell:
	docker compose exec postgres psql -U "$(POSTGRES_USER)" -d "$(POSTGRES_DB)"

migrate:
	yoyo apply --database "$(YOYO_DATABASE_URL)"

rollback:
	yoyo rollback --database "$(YOYO_DATABASE_URL)"
