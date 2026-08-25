# Deploy na AWS — ECS Fargate + ALB

> Use o template `cloudformation.yaml` na raiz do projeto.
> **Importante:** treine os modelos **antes** do build Docker (`make train`).

## Pré-requisitos

- AWS CLI configurado (`aws configure`)
- Docker instalado
- Permissões: ECR, ECS, CloudFormation, IAM

### Múltiplos profiles AWS

Se você tem mais de um profile na máquina (ex.: conta pessoal e conta da FIAP), defina qual usar **antes** dos comandos `aws`:

```bash
# listar profiles disponíveis
# AWS CLI v2:
aws configure list-profiles

# AWS CLI v1 (alternativa compatível):
grep '^\[' ~/.aws/credentials 2>/dev/null | tr -d '[]'
grep '^\[profile ' ~/.aws/config 2>/dev/null | sed 's/\[profile \(.*\)\]/\1/'

# usar um profile específico nesta sessão de terminal
export AWS_PROFILE=seu-profile

# confirmar conta e região ativas
aws sts get-caller-identity
```

Alternativa sem variável de ambiente: adicione `--profile seu-profile` em cada comando `aws`.

> Os exemplos abaixo assumem que `AWS_PROFILE` já foi exportado (ou que você usa o profile `default`).

**`list-profiles` ainda falha após `brew install awscli`?** O pyenv/pip pode estar
priorizando um AWS CLI v1 antigo. Confira:

```bash
which aws && aws --version
# se mostrar pyenv/shims e aws-cli/1.x, use o v2 do Homebrew:
/opt/homebrew/bin/aws configure list-profiles
# ou remova o v1: pip uninstall awscli
```

## Passo 1 — Treinar artefatos

```bash
make download-data
make train
ls models/   # deve conter triage_pipeline.joblib, triage_pipeline.onnx, metrics.json
```

## Passo 2 — Push para ECR

> **Zsh:** use `${VAR}:latest` (com chaves). Sem chaves, `$ECR_REPO:latest` vira
> `medical-triage-apiatest` porque o zsh interpreta `:l` como modificador lowercase.

```bash
# opcional, se ainda não exportou no início
# export AWS_PROFILE=seu-profile

AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
AWS_REGION=us-east-1
ECR_REPO=medical-triage-api

aws ecr create-repository --repository-name $ECR_REPO --region $AWS_REGION 2>/dev/null || true

aws ecr get-login-password --region $AWS_REGION | \
  docker login --username AWS --password-stdin \
  $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com

docker build --platform linux/amd64 -t ${ECR_REPO}:latest .
docker tag ${ECR_REPO}:latest \
  ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPO}:latest
docker push ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPO}:latest
```

> **Mac Apple Silicon (M1/M2/M3):** o ECS Fargate usa **linux/amd64**. Sem
> `--platform linux/amd64`, a task falha com `exec format error` nos logs.

## Passo 3 — CloudFormation

```bash
aws cloudformation deploy \
  --template-file cloudformation.yaml \
  --stack-name medical-triage-prod \
  --parameter-overrides \
    ImageUri=${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPO}:latest \
    ApiKey=$(openssl rand -hex 16) \
  --capabilities CAPABILITY_NAMED_IAM \
  --region $AWS_REGION
```

## Passo 4 — Obter URL

```bash
aws cloudformation describe-stacks \
  --stack-name medical-triage-prod \
  --query "Stacks[0].Outputs[?OutputKey=='LoadBalancerDNS'].OutputValue" \
  --output text
```

## Passo 5 — Testar

```bash
ALB_DNS=<saida-do-passo-anterior>
API_KEY=<sua-chave>

curl http://$ALB_DNS/health

curl -X POST http://$ALB_DNS/predict \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $API_KEY" \
  -d '{"text": "Acute myocardial infarction with ST elevation and elevated troponin."}'
```

## HTTPS (recomendado)

1. Solicite certificado no ACM (mesma região do ALB).
2. Adicione listener HTTPS (443) no ALB apontando para o Target Group.
3. Redirecione HTTP → HTTPS.

## Alternativa: App Runner

Para deploy mais rápido (sem VPC manual), siga o padrão da Fase 1 em
`docs/aws_deploy.md` da fase 1 — substitua a imagem e ajuste a porta 8000.

## Limpeza

```bash
# use o mesmo AWS_PROFILE da sessão de deploy
aws cloudformation delete-stack --stack-name medical-triage-prod
aws ecr delete-repository --repository-name $ECR_REPO --force
```

## Troubleshooting ECS

| Sintoma | Causa provável | Solução |
|---------|----------------|---------|
| Task para com `exec format error` | Imagem ARM64 no Mac, Fargate x86_64 | `docker build --platform linux/amd64 ...` e novo push |
| Task reinicia em loop | Health check falha (`model_loaded: false`) | Confirme `models/*.onnx` no build (`make train`) |
| `CannotPullContainerError` | Imagem não existe no ECR ou URI errada | Verifique `ImageUri` no CloudFormation |

Após corrigir a imagem, force novo deploy:

```bash
aws ecs update-service \
  --cluster medical-triage-cluster \
  --service medical-triage-service \
  --force-new-deployment \
  --region ${AWS_REGION}
```

Ver logs da task:

```bash
aws logs tail /ecs/medical-triage --follow --region ${AWS_REGION}
```
