# 📊 Portfolio Management Assistant

Built this tool (POC) for an investment firm that sell model portfolios for easier and smarter onboarding of assets.

## ✨ Features

### 🎯 **Core Portfolio Management**
- **Add/Remove Tickers**: Easily add or remove stocks, ETFs, and other securities
- **Weight Management**: Set and modify allocation percentages with automatic validation
- **Portfolio Rebalancing**: AI-powered rebalancing suggestions and execution
- **Real-time Validation**: Prevents over-allocation and ensures portfolio integrity

### 🏷️ **Smart Tagging & Classification**
- **Multi-dimensional Tagging**: Classify tickers by asset class, instrument type, sector, region, risk level, and more
- **AI-Powered Classification**: Automatic classification using financial knowledge
- **Custom Tag Definitions**: Define your own classification categories and values
- **Bulk Operations**: Apply tags to multiple tickers at once

### 💬 **Natural Language Interface**
- **Chat with AI**: Communicate with your portfolio using natural language
- **Intelligent Commands**: The AI understands various ways to express the same request
- **Context Awareness**: AI remembers your portfolio state and preferences
- **Real-time Updates**: Changes are reflected immediately in the interface

### 📈 **Visual Portfolio Management**
- **Interactive Table**: Click to edit tag values directly in the table
- **Color-coded Tags**: Visual distinction between different tag types
- **Weight Summary**: Real-time tracking of total allocation and remaining capacity
- **Responsive Design**: Works seamlessly on desktop and mobile devices

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- OpenAI API key

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd portfolioagent5
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   Create a `.env` file in the project root:
   ```env
   OPENAI_API_KEY=your_openai_api_key_here
   ```

4. **Run the application**
   ```bash
   python main.py
   ```

5. **Open your browser**
   Navigate to `http://localhost:8000`

## 💡 How to Use

### 🎯 **Basic Portfolio Operations**

#### Adding Tickers
```
"Add AAPL with 30% weight"
"Add SPY, QQQ, VTI with 25% each"
"Add Microsoft and Google with 20% and 15%"
```

#### Managing Weights
```
"Change AAPL to 35%"
"Reduce SPY to 20%"
"Rebalance to equal weights"
```

#### Removing Tickers
```
"Remove AAPL from my portfolio"
"Delete Microsoft"
"Take out all tech stocks"
```

### 🏷️ **Tagging and Classification**

#### Setting Up Tags
```
"Tag asset classes (equity vs fixed income vs alternative)"
"Add sector tags (technology, healthcare, financial services)"
"Create region tags (US, Europe, Asia-Pacific)"
```

#### Classifying Tickers
```
"Classify by asset class"
"Add instrument type tags"
"Tag by sector and region"
```

#### Editing Tag Values
**Via Chat:**
```
"Change SPY's asset class to Fixed Income"
"Update AAPL's sector to Technology"
"Set all ETFs to Mutual Fund for instrument type"
"Make VTI's region US"
```

**Via Table:**
- Click on any tag value in the table to edit it directly
- Press Enter to save, Escape to cancel
- Changes are saved automatically

### 📊 **Portfolio Analysis**

#### Getting Portfolio Information
```
"Show my portfolio"
"What's my current allocation?"
"Display portfolio summary"
"How is my portfolio looking?"
```

#### Weight Analysis
```
"What's my total allocation?"
"How much weight is remaining?"
"Am I over-allocated?"
```

## 🎨 **User Interface Guide**

### 📋 **Portfolio Table**
- **Symbol Column**: Ticker symbols
- **Weight Column**: Allocation percentages
- **Tag Columns**: Dynamic columns for each tag type
- **Actions Column**: Edit and delete buttons

### 💬 **Chat Interface**
- **Message Input**: Type your commands here
- **Send Button**: Submit your message
- **Chat History**: View conversation history
- **AI Responses**: Get intelligent feedback and confirmations

### ⚙️ **Control Panel**
- **New Session**: Start fresh with empty portfolio
- **Save Tags**: Save current tag configuration
- **Load Tags**: Load previously saved configurations

## 🔧 **Advanced Features**

### 🏷️ **Tag Management**

#### Creating Custom Tags
1. Use the chat to define tag categories:
   ```
   "Tag risk levels (low, moderate, high, very high)"
   ```

2. The system will create the tag structure
3. Classify your tickers using the new tags

#### Bulk Tag Operations
```
"Set all ETFs to Mutual Fund for instrument type"
"Change all Equity to Fixed Income for asset class"
"Update all US to Global for region"
```

### 📊 **Portfolio Optimization**

#### Rebalancing
```
"Rebalance my portfolio"
"Make all weights equal"
"Adjust weights to 40% stocks, 40% bonds, 20% alternatives"
```

#### Weight Validation
- Automatic validation prevents over-allocation
- Real-time feedback on weight changes
- Visual indicators for allocation status

## 🎯 **Example Workflows**

### 📈 **Building a Diversified Portfolio**

1. **Add Core Holdings**
   ```
   "Add SPY with 40%, VTI with 30%, BND with 20%"
   ```

2. **Classify by Asset Class**
   ```
   "Classify by asset class"
   ```

3. **Add Sector Diversification**
   ```
   "Add QQQ with 10%"
   "Classify by sector"
   ```

4. **Fine-tune Allocations**
   ```
   "Reduce SPY to 35%, increase VTI to 35%"
   ```

### 🏷️ **Comprehensive Tagging**

1. **Set Up Multiple Tag Types**
   ```
   "Tag asset classes (equity, fixed income, alternative)"
   "Add instrument types (ETF, mutual fund, stock, bond)"
   "Create sector tags (technology, healthcare, financial)"
   ```

2. **Classify All Tickers**
   ```
   "Classify by all available tags"
   ```

3. **Edit Individual Values**
   - Click on table cells to edit directly
   - Or use chat commands for specific changes

## 🛠️ **Technical Details**

### 🏗️ **Architecture**
- **Backend**: Python with FastAPI and LangGraph
- **Frontend**: HTML, CSS, JavaScript
- **AI Integration**: OpenAI GPT models
- **State Management**: LocalStorage for persistence

### 🔧 **Configuration**
- **Model Selection**: Automatically uses the best available GPT model
- **Temperature**: Optimized for consistent, reliable responses
- **Tool Integration**: Seamless AI tool calling for portfolio operations

### 📱 **Browser Compatibility**
- Chrome (recommended)
- Firefox
- Safari
- Edge

## 🆘 **Troubleshooting**

### Common Issues

**Q: The AI isn't responding to my commands**
A: Make sure your OpenAI API key is correctly set in the `.env` file

**Q: I can't edit tag values in the table**
A: Click directly on the tag value cell to enter edit mode

**Q: My portfolio weights don't add up to 100%**
A: The system prevents over-allocation. Check your total weights and adjust accordingly

**Q: The AI classified my tickers incorrectly**
A: You can edit any classification by clicking on the table cell or using chat commands

### Getting Help

1. **Check the chat interface** for AI suggestions
2. **Use the help commands** like "Show my portfolio" to understand current state
3. **Try different phrasings** if a command doesn't work
4. **Use the table interface** for direct editing when needed

## 🎉 **Tips for Best Results**

### 💬 **Effective Chat Commands**
- Be specific: "Add AAPL with 30% weight" vs "Add Apple"
- Use clear tag names: "asset class" vs "type"
- Check your portfolio regularly: "Show my portfolio"

### 🏷️ **Tag Management**
- Set up tags before adding many tickers
- Use consistent naming conventions
- Edit classifications as needed for accuracy

### 📊 **Portfolio Building**
- Start with broad categories (asset classes)
- Add specific details (sectors, regions) later
- Regularly review and rebalance

## 🔮 **Future Enhancements**

- **Performance Analytics**: Track portfolio performance over time
- **Risk Analysis**: Advanced risk metrics and analysis
- **Tax Optimization**: Tax-loss harvesting suggestions
- **Integration**: Connect with real brokerage accounts
- **Reporting**: Generate detailed portfolio reports
