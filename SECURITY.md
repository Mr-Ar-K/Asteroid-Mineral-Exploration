# API Security and Configuration Guide

## 🔐 Protecting Your API Keys

This project uses NASA APIs for asteroid data retrieval. To keep your API keys secure:

### 1. Environment Variables (Recommended)

Set your NASA API key as an environment variable:

```bash
# Linux/Mac
export NASA_API_KEY="your_actual_api_key_here"

# Windows (Command Prompt)
set NASA_API_KEY=your_actual_api_key_here

# Windows (PowerShell)
$env:NASA_API_KEY="your_actual_api_key_here"
```

### 2. .env File (Alternative)

Create a `.env` file in the project root (this file is automatically ignored by git):

```bash
# Copy the template
cp .env.template .env

# Edit with your actual values
NASA_API_KEY=your_actual_api_key_here
NASA_EMAIL=your_email@example.com
NASA_ACCOUNT_ID=your_account_id_here
```

### 3. Getting Your NASA API Key

1. Visit [NASA API Portal](https://api.nasa.gov/)
2. Click "Generate API Key" 
3. Fill out the form with your information
4. Use the provided API key in your environment variables

## 🚨 Security Best Practices

### What's Protected
- ✅ API keys are loaded from environment variables
- ✅ Sensitive config files are in `.gitignore`
- ✅ Template files show structure without secrets
- ✅ Configuration system prioritizes environment variables

### Files to Never Commit
- `config/config.yaml` (if it contains real API keys)
- `.env` files
- Any file with actual API keys or passwords

### Configuration Hierarchy
1. **Environment Variables** (highest priority)
2. **config.yaml** (if exists)
3. **config.template.yaml** (fallback)
4. **Minimal defaults** (if no config found)

## 🔧 Project Setup

### For Development

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd Asteroid-Mineral-Exploration
   ```

2. **Set up your API key**
   ```bash
   # Option A: Environment variable
   export NASA_API_KEY="your_api_key_here"
   
   # Option B: Create .env file
   cp .env.template .env
   # Edit .env with your real API key
   ```

3. **Create your config file (optional)**
   ```bash
   cp config/config.template.yaml config/config.yaml
   # Edit config.yaml if you need custom settings
   ```

4. **Run the project**
   ```bash
   python launcher.py --dashboard
   ```

### For Production

Use environment variables exclusively in production:

```bash
# Docker example
docker run -e NASA_API_KEY="your_key" your-asteroid-app

# Kubernetes example
kubectl create secret generic nasa-api-key \
  --from-literal=NASA_API_KEY="your_key"
```

## ⚠️ If You Accidentally Committed Secrets

If you accidentally committed API keys to git:

1. **Immediately rotate the API key** at [NASA API Portal](https://api.nasa.gov/)
2. **Remove from git history**:
   ```bash
   # Remove the file from git history
   git filter-branch --force --index-filter \
     "git rm --cached --ignore-unmatch config/config.yaml" \
     --prune-empty --tag-name-filter cat -- --all
   
   # Force push (⚠️ WARNING: This rewrites history)
   git push --force --all
   ```
3. **Add to .gitignore** (already done in this project)
4. **Update all team members** to pull the cleaned repository

## 📝 Environment Variables Reference

| Variable | Description | Required | Example |
|----------|-------------|----------|---------|
| `NASA_API_KEY` | NASA API key for asteroid data | Yes | `ieNKM2I1HjxFt...` |
| `NASA_EMAIL` | Your NASA account email | No | `user@example.com` |
| `NASA_ACCOUNT_ID` | Your NASA account ID | No | `uuid-string` |
| `LOG_LEVEL` | Logging level | No | `INFO` |
| `DATA_CACHE_DIR` | Cache directory path | No | `data/cache` |

## 🛡️ Additional Security Measures

1. **API Rate Limiting**: Built-in rate limiting prevents API abuse
2. **Request Monitoring**: All API calls are logged for monitoring
3. **Error Handling**: API errors don't expose sensitive information
4. **Session Management**: HTTP sessions are properly configured

## 📞 Support

If you have issues with API configuration:
1. Check the logs in `logs/` directory
2. Verify your API key at [NASA API Portal](https://api.nasa.gov/)
3. Ensure environment variables are properly set
4. Review this security guide
