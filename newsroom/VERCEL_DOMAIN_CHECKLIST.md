# Vercel + Custom Domain Checklist

## 1) Repository
- Keep the static site under `web/`.
- Ensure `web/vercel.json` is committed.

## 2) Connect Project in Vercel
1. Create a new project in Vercel.
2. Import this GitHub repository.
3. Framework preset: `Other`.
4. Root directory: `web`.
5. Build command: leave empty for static hosting.
6. Output directory: leave empty.

## 3) Continuous Deploy
- Enable auto deploy for branch `main`.
- Every push triggers a new deployment.

## 4) Attach Domain
1. Open Vercel project settings.
2. Add your domain in `Domains`.
3. Copy DNS records shown by Vercel.
4. Add those records in your DNS provider.
5. Wait for propagation.

## 5) Fresh Data Behavior
- `web/vercel.json` can force `no-store` for `/data/*` so readers always receive fresh feed JSON.
