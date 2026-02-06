# Vercel + Dominio propio (Checklist)

## 1) Repo de contenido
- El sitio está en: `/Users/c/Library/LaAurora/web`
- El push horario ya está automatizado con `com.la-aurora.github-sync`.

## 2) Conectar repo en Vercel
1. Entra a Vercel > `Add New Project`.
2. Importa el repo GitHub donde se publica `/Users/c/Library/LaAurora/web`.
3. Framework Preset: `Other`.
4. Root Directory: `/` (raíz del repo web).
5. Build Command: vacío.
6. Output Directory: vacío.

## 3) Deploy continuo
- Deja activado `Auto Deploy` en rama `main`.
- Cada push del job horario generará un nuevo deploy.

## 4) Dominio propio
1. En Vercel Project > `Settings` > `Domains`, añade tu dominio.
2. Copia los registros DNS que te da Vercel.
3. En tu registrador (Cloudflare, Namecheap, etc.) crea esos registros.
4. Espera propagación DNS (normalmente minutos, a veces hasta 24h).

## 5) Cache de datos vivos
- `web/vercel.json` ya fuerza `no-store` en `/data/*` para que portada y artículos nuevos refresquen en cada visita.
