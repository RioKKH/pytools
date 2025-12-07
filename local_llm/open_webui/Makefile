.PHONY: start stop restart logs clean

# 起動
start:
	docker compose up -d
	@echo "Open WebUI: http://localhost:3000"

# 停止
stop:
	docker compose down

# 再起動
restart:
	docker compose restart

# ログ表示
logs:
	docker compose logs -f

# ボリュームも含めて削除
clean:
	docker compose down -v
