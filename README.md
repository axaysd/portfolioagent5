# 🚀 Portfolio Management with LangGraph AI

A modern portfolio management application powered by **LangGraph** and **OpenAI's latest models** (including GPT-5 when available). This application provides an intuitive interface for managing investment portfolios with AI-powered insights and natural language interaction.

## ✨ Features

- **🤖 AI-Powered Portfolio Management**: Built with LangGraph for intelligent portfolio operations
- **💬 Natural Language Interface**: Chat with your portfolio using natural language
- **📊 Real-time Portfolio Tracking**: Monitor weights, allocations, and portfolio health
- **🎯 Smart Validation**: Ensures portfolio weights never exceed 100%
- **📱 Responsive Design**: Modern, responsive UI that works on all devices
- **🔄 Easy Model Switching**: Support for GPT-4o, GPT-4o-mini, and GPT-5 (when available)

## 🏗️ Architecture

- **Frontend**: Modern HTML5, CSS3, and Vanilla JavaScript
- **Backend**: FastAPI with Python
- **AI Engine**: LangGraph with OpenAI integration
- **State Management**: Client-side with localStorage persistence
- **Responsive Design**: CSS Grid and Flexbox for modern layouts

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- OpenAI API key
- Modern web browser

### 1. Clone and Install

```bash
git clone <repository-url>
cd portfolioagent5
uv pip install -r requirements.txt
```

### 2. Configure OpenAI API

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_actual_openai_api_key_here
OPENAI_MODEL=gpt-5
```

**Get your API key from**: [OpenAI Platform](https://platform.openai.com/api-keys)

**Note**: GPT-5 is now the default model for the best performance and capabilities!

### 3. Run the Application

```bash
python main.py
```

The application will start at `http://localhost:8000`

## 🎯 Usage Examples

### Portfolio Management via Chat

You can interact with your portfolio using natural language:

- **"Add AAPL with 30% weight"**
- **"Remove GOOGL from my portfolio"**
- **"What's in my portfolio?"**
- **"How is my portfolio looking?"**
- **"Modify MSFT to 25% weight"**

### Manual Portfolio Management

- Use the input fields to add tickers and weights
- Edit existing positions with the pencil icon
- Delete positions with the trash icon
- Real-time validation ensures weights stay under 100%

## 🤖 AI Models Supported

| Model | Description | Use Case |
|-------|-------------|----------|
| **GPT-5** ⭐ | Latest and most capable - best for coding and agentic tasks | Production, complex analysis, portfolio optimization |
| **GPT-5 Mini** | Faster, cost-efficient version for well-defined tasks | Development, testing, standard operations |
| **GPT-5 Nano** | Fastest, most cost-efficient for summarization and classification | Quick portfolio summaries, cost-sensitive operations |
| **GPT-5 Chat** | Used in ChatGPT with reasoning capabilities | Conversational portfolio management |
| **GPT-4o** | Previous generation - still very capable | Legacy compatibility, fallback option |
| **GPT-4o-mini** | Faster, more affordable | Development, testing with GPT-4 capabilities |
| **GPT-4-turbo** | Previous generation | Legacy compatibility |

### Switching Models

You can switch models via the API:

```bash
# Switch to GPT-5 Mini for faster, cost-efficient operations
curl -X POST http://localhost:8000/api/update-model \
  -H "Content-Type: application/json" \
  -d '{"model": "gpt-5-mini"}'

# Switch to GPT-5 Nano for maximum speed and cost efficiency
curl -X POST http://localhost:8000/api/update-model \
  -H "Content-Type: application/json" \
  -d '{"model": "gpt-5-nano"}'

# Switch back to full GPT-5 for maximum capabilities
curl -X POST http://localhost:8000/api/update-model \
  -H "Content-Type: application/json" \
  -d '{"model": "gpt-5"}'
```

**Default Model**: The application now uses **GPT-5** by default for the best performance and capabilities!

## 🔧 API Endpoints

### Chat Interface
- `POST /api/chat` - Send messages to the AI agent
- `GET /api/model-info` - Get current model information
- `POST /api/update-model` - Switch AI models
- `GET /api/health` - Health check

### Frontend
- `GET /` - Main portfolio management interface
- `GET /static/*` - Static assets (CSS, JS)

## 🏗️ LangGraph Integration

The application uses LangGraph to create an intelligent workflow:

1. **Request Analysis**: AI analyzes user input to determine intent
2. **Action Execution**: Performs portfolio operations (add/remove/modify)
3. **Response Generation**: Creates natural language responses
4. **State Management**: Maintains conversation and portfolio state

### LangGraph Nodes

- `analyze_request`: Determines user intent and action
- `execute_action`: Performs portfolio operations
- `generate_response`: Creates AI responses

## 🎨 UI/UX Features

- **Modern Design**: Clean, professional interface with gradients and shadows
- **Responsive Layout**: 2/3 left panel for portfolio, 1/3 right for chat
- **Subtle Actions**: Edit/delete buttons appear on hover
- **Real-time Feedback**: Instant validation and notifications
- **Smooth Animations**: CSS transitions for better user experience

## 🔒 Security & Validation

- **Input Validation**: Client-side validation for ticker symbols and weights
- **Weight Limits**: Ensures portfolio never exceeds 100% allocation
- **Duplicate Prevention**: Prevents adding the same ticker twice
- **API Key Protection**: Environment variable configuration

## 🚀 Deployment

### Local Development
```bash
python main.py
```

### Production Deployment
```bash
# Install production dependencies
uv pip install -r requirements.txt

# Set environment variables
export OPENAI_API_KEY=your_key
export OPENAI_MODEL=gpt-4o

# Run with production server
uvicorn main:app --host 0.0.0.0 --port 8000
```

### Docker (Optional)
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN uv pip install -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

## 🧪 Testing

### Test the LangGraph Agent

Test the LangGraph agent directly:

```bash
python langgraph_model.py
```

This will run a series of test conversations to verify the agent's functionality.

### Test GPT-5 Integration

Verify that GPT-5 is working correctly:

```bash
python test_gpt5.py
```

This comprehensive test suite will:
- ✅ Verify OpenAI API key configuration
- ✅ Test LangGraph model import and creation
- ✅ Test GPT-5 chat functionality
- ✅ Test model switching between GPT-5 variants
- ✅ Validate portfolio management operations

**Note**: Make sure you have set your `OPENAI_API_KEY` in the `.env` file before running tests.

## 🔧 Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | Required | Your OpenAI API key |
| `OPENAI_MODEL` | `gpt-5` | AI model to use (now defaults to GPT-5!) |

**Available GPT-5 Models:**
- `gpt-5`: Full GPT-5 model (default) - best for complex tasks
- `gpt-5-mini`: Faster, cost-efficient version
- `gpt-5-nano`: Fastest, most cost-efficient
- `gpt-5-chat`: ChatGPT-optimized version

### Customization

- **Models**: Add new models in `config.py`
- **Styling**: Modify `static/styles.css` for UI changes
- **Logic**: Extend `langgraph_model.py` for new AI capabilities

## 🐛 Troubleshooting

### Common Issues

1. **"No OpenAI API key configured"**
   - Create a `.env` file with your API key
   - Or set `OPENAI_API_KEY` environment variable

2. **"Error processing chat"**
   - Check your internet connection
   - Verify API key is valid
   - Check server logs for detailed errors

3. **Portfolio not updating**
   - Refresh the page
   - Check browser console for JavaScript errors
   - Verify localStorage is enabled

### Debug Mode

Enable detailed logging by modifying `main.py`:

```python
uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, log_level="debug")
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **LangGraph**: For the powerful orchestration framework
- **OpenAI**: For the advanced language models
- **FastAPI**: For the modern Python web framework
- **Community**: For feedback and contributions

## 📞 Support

- **Issues**: Create an issue on GitHub
- **Discussions**: Use GitHub Discussions for questions
- **Documentation**: Check the inline code comments

---

**Happy Portfolio Management! 🎯📈**