# La Aurora Newsroom

Generador editorial automatizado para `LaAurora`, con:
- redacción IA 24/7,
- búsqueda web + RSS,
- datos de mercado en tiempo real (cripto/bolsa),
- enrutado multi-Mac por rol LM Studio,
- publicación web estática en `web/data/articles.json`.

## 1) Ejecución local

```bash
cd /Users/c/Library/LaAurora/newsroom
python3 ai_publisher.py --config config.json --topic "ibex 35 y bitcoin" --region world --min-words 1200
```

Salida:
- actualiza `/Users/c/Library/LaAurora/web/data/articles.json`

## 2) Clúster de 4 Macs (LM Studio)

El pipeline admite rutas por rol (`chief`, `research`, `fact`, `tagger`, `embedding`) en `config.json`.

Plantilla:
- `routes.4macs.example.json`
- `config.example.json` (incluye ejemplo completo por IP)

Autodetección de nodos LM Studio:

```bash
cd /Users/c/Library/LaAurora/newsroom
python3 discover_lm_workers.py --json
```

Autoconfiguración por modelos (sin escribir IPs a mano):

```bash
cd /Users/c/Library/LaAurora/newsroom
python3 auto_configure_cluster.py --config config.json
```

Bootstrap completo (autoconfigura rutas + valida + reinicia pipeline + refresca sync GitHub):

```bash
cd /Users/c/Library/LaAurora/newsroom
bash bootstrap_cluster.sh config.json
```

Aplicar IPs reales en un comando:

```bash
cd /Users/c/Library/LaAurora/newsroom
python3 configure_cluster_routes.py \
  --config config.json \
  --chief-ip 10.211.0.241 \
  --research-ip 10.211.0.242 \
  --fact-ip 10.211.0.243 \
  --tagger-ip 10.211.0.244 \
  --embedding-ip 10.211.0.244
```

Comprobar conectividad de todos los nodos:

```bash
cd /Users/c/Library/LaAurora/newsroom
bash check_lm_cluster.sh config.json
```

Nota:
- Si un worker falla, el sistema hace fallback automático a rutas alternativas.
- Si solo hay 1 o 2 workers activos, el sistema reasigna varios roles al mismo nodo.

## 3) Daemon editorial 24/7

```bash
cd /Users/c/Library/LaAurora/newsroom
/bin/bash daemon.sh
```

Logs:
- `/Users/c/Library/LaAurora/newsroom/logs/publisher.log`

## 4) Watchdog y salud 24/7

Monitor automático:
- `watchdog_monitor.py`
- `com.la-aurora.watchdog.plist`

Qué hace cada 120 segundos:
- valida que `com.la-aurora.publisher` esté corriendo,
- mata procesos `ai_publisher.py` colgados (>900s),
- limpia `.runner.lock` estancado,
- comprueba frescura de `web/data/articles.json`,
- comprueba conectividad/carga de modelo por rol (`chief`, `research`, `fact`, `tagger`, `embedding`),
- registra estado en `logs/health.json`,
- registra alertas deduplicadas en `logs/health.alerts.log`.

Instalación `launchd`:

```bash
cp /Users/c/Library/LaAurora/newsroom/com.la-aurora.watchdog.plist ~/Library/LaunchAgents/com.la-aurora.watchdog.plist
launchctl bootout gui/$(id -u) com.la-aurora.watchdog >/dev/null 2>&1 || true
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.la-aurora.watchdog.plist
launchctl enable gui/$(id -u)/com.la-aurora.watchdog
launchctl kickstart -k gui/$(id -u)/com.la-aurora.watchdog
```

## 5) Push horario a GitHub

Script:
- `github_hourly_sync.sh`

Configurar credenciales/remoto:
1. Copia `github-sync.env.example` a `github-sync.env`
2. Ajusta `GITHUB_REMOTE_URL` (HTTPS+token o SSH)

Instalar `launchd` (cada hora):

```bash
cd /Users/c/Library/LaAurora/newsroom
bash setup_github_sync_launchd.sh
```

Logs del sync:
- `/Users/c/Library/LaAurora/newsroom/logs/github.sync.out.log`
- `/Users/c/Library/LaAurora/newsroom/logs/github.sync.err.log`

## 6) Vercel y dominio propio

El sitio está listo para estático con:
- `/Users/c/Library/LaAurora/web/vercel.json`

Flujo recomendado:
1. Conecta el repo en Vercel (Framework: Other, Output: raíz del repo web).
2. Activa deploy en push de rama `main`.
3. Añade tu dominio en Vercel Project > Domains.
4. En tu proveedor DNS crea los registros que indique Vercel.
