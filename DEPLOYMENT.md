# 🚀 Deployment Guide for Portfolio Agent

This guide will help you deploy your Portfolio Agent application to the web so users can test it.

## 📋 Prerequisites

1. **OpenAI API Key**: You need an OpenAI API key to use the AI features
2. **Git Repository**: Your code should be in a Git repository (GitHub, GitLab, etc.)
3. **Environment Variables**: Set up your API keys securely

## 🌐 Deployment Options

### Option 1: Render (Recommended - FREE)

**Step 1: Prepare Your Repository**
- Push your code to GitHub/GitLab
- Ensure all files are committed

**Step 2: Deploy on Render**
1. Go to [render.com](https://render.com) and sign up
2. Click "New +" → "Web Service"
3. Connect your GitHub/GitLab repository
4. Configure the service:
   - **Name**: `portfolio-agent-app`
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`
5. Add environment variables:
   - `OPENAI_API_KEY`: Your OpenAI API key
   - `OPENAI_MODEL`: `gpt-5-mini` (or your preferred model)
6. Click "Create Web Service"

**Step 3: Wait for Deployment**
- Render will automatically build and deploy your app
- You'll get a URL like: `https://your-app-name.onrender.com`

### Option 2: Railway (Fast & Easy)

**Step 1: Deploy on Railway**
1. Go to [railway.app](https://railway.app) and sign up
2. Click "New Project" → "Deploy from GitHub repo"
3. Select your repository
4. Railway will auto-detect it's a Python app
5. Add environment variables:
   - `OPENAI_API_KEY`: Your OpenAI API key
   - `PORT`: `8000`

**Step 2: Get Your URL**
- Railway will provide a URL like: `https://your-app-name.railway.app`

### Option 3: Fly.io (Global Deployment)

**Step 1: Install Fly CLI**
```bash
# Windows (PowerShell)
iwr https://fly.io/install.ps1 -useb | iex

# Or download from: https://fly.io/docs/hands-on/install-flyctl/
```

**Step 2: Deploy**
```bash
fly auth login
fly launch
# Follow the prompts, select your app name
fly deploy
```

### Option 4: Heroku (Paid but Easy)

**Step 1: Install Heroku CLI**
```bash
# Download from: https://devcenter.heroku.com/articles/heroku-cli
```

**Step 2: Deploy**
```bash
heroku login
heroku create your-app-name
git push heroku main
heroku config:set OPENAI_API_KEY=your_api_key_here
heroku open
```

## 🔧 Environment Variables

Set these in your deployment platform:

```bash
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-5-mini
PORT=8000
```

## 🚨 Important Security Notes

1. **Never commit your API keys** to Git
2. **Use environment variables** for sensitive data
3. **Enable HTTPS** (most platforms do this automatically)
4. **Set up CORS** properly if needed

## 📱 Testing Your Deployed App

Once deployed, users can:

1. **Access the web interface** at your deployment URL
2. **Use the chat functionality** to interact with the portfolio agent
3. **Test portfolio management** features
4. **View API documentation** at `/docs` endpoint

## 🔍 Troubleshooting

### Common Issues:

1. **Build Failures**:
   - Check Python version compatibility
   - Ensure all dependencies are in `requirements.txt`

2. **Runtime Errors**:
   - Check environment variables are set correctly
   - Verify API keys are valid

3. **Port Issues**:
   - Use `$PORT` environment variable (platforms set this automatically)
   - Don't hardcode port numbers

### Debug Commands:

```bash
# Check logs (Render)
render logs

# Check logs (Railway)
railway logs

# Check logs (Fly.io)
fly logs

# Check logs (Heroku)
heroku logs --tail
```

## 🌟 Next Steps After Deployment

1. **Test all functionality** thoroughly
2. **Set up monitoring** and logging
3. **Configure custom domain** if desired
4. **Set up CI/CD** for automatic deployments
5. **Monitor usage** and costs

## 💰 Cost Considerations

- **Render**: Free tier (750 hours/month)
- **Railway**: $5 credit monthly
- **Fly.io**: Free tier (3 VMs)
- **Heroku**: $7/month minimum

## 🆘 Need Help?

- Check platform-specific documentation
- Look at deployment logs for errors
- Ensure all files are properly committed
- Verify environment variables are set correctly

---

**Happy Deploying! 🚀**

Your Portfolio Agent will be accessible to users worldwide once deployed!
