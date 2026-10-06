from pathlib import Path
import unittest
import yaml

ROOT = Path(__file__).resolve().parents[1]


class FrontendDeploymentTests(unittest.TestCase):
    def test_frontend_build_is_locked_and_only_dist_reaches_runtime(self):
        dockerfile = (ROOT / 'frontend/Dockerfile').read_text(encoding='utf-8')
        self.assertIn('FROM node:20-alpine AS build', dockerfile)
        self.assertIn('RUN npm ci', dockerfile)
        self.assertNotIn('npm install', dockerfile)
        runtime = dockerfile.split('FROM nginx:alpine AS runtime')[1]
        self.assertIn('COPY --from=build /frontend/dist', runtime)
        self.assertNotIn('node_modules', runtime)
        self.assertNotIn('COPY src', runtime)
        self.assertIn('ENV VITE_API_BASE_URL=""', dockerfile)

    def test_frontend_has_no_secret_environment_or_build_args(self):
        compose = yaml.safe_load((ROOT / 'docker-compose.yml').read_text(encoding='utf-8'))
        frontend = compose['services']['frontend']
        self.assertNotIn('environment', frontend)
        self.assertNotIn('args', frontend['build'])
        dockerfile = (ROOT / 'frontend/Dockerfile').read_text(encoding='utf-8')
        for name in ('REVIEW_ADMIN_TOKEN', 'SILICONFLOW_API_KEY', 'DATABASE_URL'):
            self.assertNotIn(name, dockerfile)
        ignore = (ROOT / 'frontend/.dockerignore').read_text(encoding='utf-8')
        self.assertIn('.env', ignore)
        self.assertIn('node_modules', ignore)

    def test_nginx_preserves_paths_headers_and_spa_fallback(self):
        nginx = (ROOT / 'frontend/nginx.conf').read_text(encoding='utf-8')
        for route in ('/api/', '/internal/', '= /health', '= /ready'):
            self.assertIn('location ' + route, nginx)
        self.assertIn('location = /api/agent/chat/stream', nginx)
        self.assertIn('proxy_http_version 1.1;', nginx)
        self.assertIn('proxy_buffering off;', nginx)
        self.assertIn('proxy_cache off;', nginx)
        self.assertIn('proxy_read_timeout 180s;', nginx)
        self.assertEqual(nginx.count('proxy_pass http://api:8000;'), 5)
        self.assertIn('try_files $uri $uri/ /index.html;', nginx)
        for header in ('Host', 'X-Real-IP', 'X-Forwarded-For', 'X-Forwarded-Proto'):
            self.assertIn('proxy_set_header ' + header, nginx)
        self.assertNotIn('proxy_set_header X-Review-Token', nginx)

    def test_existing_start_order_and_local_ports_remain(self):
        services = yaml.safe_load((ROOT / 'docker-compose.yml').read_text(encoding='utf-8'))['services']
        self.assertEqual(set(services), {'postgres', 'db-init', 'api', 'frontend'})
        self.assertEqual(services['api']['depends_on']['db-init']['condition'], 'service_completed_successfully')
        self.assertEqual(services['db-init']['depends_on']['postgres']['condition'], 'service_healthy')
        self.assertEqual(services['frontend']['depends_on']['api']['condition'], 'service_healthy')
        self.assertEqual(services['frontend']['ports'], ['127.0.0.1:${FRONTEND_HOST_PORT:-8080}:80'])
        self.assertIn('/ready', ' '.join(services['api']['healthcheck']['test']))
        self.assertIn('127.0.0.1/', (ROOT / 'frontend/Dockerfile').read_text(encoding='utf-8'))

if __name__ == '__main__':
    unittest.main()
