include .env
export

migrate:
	yoyo apply --database "${DATABASE_URL}"

rollback:
	yoyo rollback --database "${DATABASE_URL}"

