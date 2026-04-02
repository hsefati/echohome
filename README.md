# EcoHome Energy Advisor

An AI-powered energy optimization agent that helps customers reduce electricity costs and environmental impact through personalized recommendations.

## 📋 Project Overview

EcoHome is a smart-home energy start-up that helps customers with solar panels, electric vehicles, and smart thermostats optimize their energy usage. The Energy Advisor agent provides personalized recommendations about when to run devices to minimize costs and carbon footprint.

## 🎯 Quick Start

### Prerequisites
- Python 3.10+
- `uv` package manager (recommended) or `pip`

### Installation

**Using uv (recommended):**
```bash
uv sync
```

**Or using pip:**
```bash
pip install -r requirements.txt
```

### Environment Setup

Create a `.env` file in the root directory with your API keys:

```bash
VOCAREUM_API_KEY=your_vocareum_api_key_here
VOCAREUM_BASE_URL=https://openai.vocareum.com/v1
OPENWEATHER_API_KEY=your_openweather_api_key_here
```

**Required Environment Variables:**
- `VOCAREUM_API_KEY`: Your Vocareum API key for LLM and embeddings
- `VOCAREUM_BASE_URL`: The Vocareum OpenAI-compatible base URL
- `OPENWEATHER_API_KEY`: Your OpenWeather API key for weather forecasts

### Running the Project

Navigate to the `ecohome/` directory and execute the notebooks in order:

```bash
cd ecohome
```

1. **01_db_setup.ipynb** - Set up the database and populate with sample data
2. **02_rag_setup.ipynb** - Configure the RAG pipeline for energy tips
3. **03_run_and_evaluate.ipynb** - Test and evaluate the agent

## 🤖 Agent Architecture

### Core Components

```
Energy Advisor Agent
├── LLM Engine (Vocareum API)
│   ├── Model: GPT-4 or compatible
│   ├── Temperature: 0.7 (balanced reasoning)
│   └── Token Limit: 4096
│
├── Tool Suite
│   ├── Weather Tools
│   │   ├── get_weather_forecast() - Hourly weather predictions
│   │   └── get_solar_irradiance() - Solar generation forecasts
│   │
│   ├── Pricing Tools
│   │   ├── get_electricity_pricing() - Time-of-day electricity rates
│   │   └── get_peak_hours() - Identify peak and off-peak times
│   │
│   ├── Database Tools
│   │   ├── query_energy_usage() - Historical consumption data
│   │   ├── query_solar_generation() - Past solar production
│   │   └── get_device_stats() - Device-specific analytics
│   │
│   ├── RAG Pipeline
│   │   ├── search_energy_tips() - Retrieve best practices
│   │   ├── get_device_recommendations() - Device-specific advice
│   │   └── get_optimization_strategies() - Energy optimization tips
│   │
│   └── Calculation Tools
│       ├── calculate_savings() - Cost/carbon savings estimates
│       └── calculate_roi() - Return on investment analysis
│
├── Memory & Context
│   ├── Conversation History
│   └── User Profile (devices, usage patterns)
│
└── Output Generation
    ├── Natural Language Response
    ├── Actionable Recommendations
    └── Savings & Impact Summary
```

### Agent Workflow

```
User Query
    ↓
Agent Receives Question
    ↓
Tool Selection & Planning
    ├── Analyze query intent
    ├── Determine required tools
    └── Plan tool execution order
    ↓
Tool Execution
    ├── Weather & Solar Data
    ├── Pricing Information
    ├── Historical Usage Data
    ├── RAG Search for Best Practices
    └── Calculate Potential Savings
    ↓
Response Generation
    ├── Synthesize findings
    ├── Generate recommendations
    └── Provide cost/carbon impact
    ↓
User Response
```

## 📁 Project Structure

```
echohome/
├── models/
│   ├── __init__.py
│   └── energy.py                    # SQLAlchemy ORM models
│
├── data/
│   ├── documents/
│   │   ├── tip_device_best_practices.txt
│   │   ├── tip_energy_savings.txt
│   │   ├── tip_energy_storage.txt
│   │   ├── tip_hvac_optimization.txt
│   │   ├── tip_renewable_integration.txt
│   │   ├── tip_seasonal_energy_management.txt
│   │   └── tip_smart_home_automation.txt
│   │
│   └── vectorstore/
│       ├── chroma.sqlite3           # ChromaDB vector storage
│       └── embeddings/
│
├── 01_db_setup.ipynb               # Database initialization
├── 02_rag_setup.ipynb              # RAG pipeline configuration
├── 03_run_and_evaluate.ipynb       # Agent testing & evaluation
│
├── agent.py                        # Main Energy Advisor agent implementation
├── tools.py                        # Tool definitions and implementations
├── utils.py                        # Utility functions
├── requirements.txt                # Python dependencies
├── eval_report.json                # Evaluation metrics and results
└── README.md                       # Detailed project documentation
```

## ✨ Agent Capabilities

### Key Features

- **Weather Integration**: Uses weather forecasts to predict solar generation
- **Dynamic Pricing**: Considers time-of-day electricity prices for cost optimization
- **Historical Analysis**: Queries past energy usage patterns for personalized advice
- **RAG Pipeline**: Retrieves relevant energy-saving tips and best practices
- **Multi-device Optimization**: Handles EVs, HVAC, appliances, and solar systems
- **Cost Calculations**: Provides specific savings estimates and ROI analysis

### Example Questions

The Energy Advisor can answer questions like:

- "When should I charge my electric car tomorrow to minimize cost and maximize solar power?"
- "What temperature should I set my thermostat on Wednesday afternoon if electricity prices spike?"
- "Suggest three ways I can reduce energy use based on my usage history."
- "How much can I save by running my dishwasher during off-peak hours?"
- "What's the best time to run my HVAC system this week based on weather and pricing?"

## 🗄️ Database Schema

### Energy Usage Table
- `timestamp`: When the energy was consumed
- `consumption_kwh`: Amount of energy used
- `device_type`: Type of device (EV, HVAC, appliance)
- `device_name`: Specific device name
- `cost_usd`: Cost at time of usage

### Solar Generation Table
- `timestamp`: When the energy was generated
- `generation_kwh`: Amount of solar energy produced
- `weather_condition`: Weather during generation
- `temperature_c`: Temperature at time of generation
- `solar_irradiance`: Solar irradiance level

## 🧠 Technology Stack

- **LangChain**: Agent framework and tool integration
- **LangGraph**: Agent orchestration and workflow
- **ChromaDB**: Vector database for document retrieval
- **SQLAlchemy**: Database ORM and management
- **Vocareum API**: LLM and embeddings (OpenAI-compatible)
- **OpenWeather API**: Weather forecasts and data
- **SQLite**: Local database storage

## 📊 Evaluation Criteria

The agent is evaluated on:

- **Accuracy**: Correct information and calculations
- **Relevance**: Responses address the user's question
- **Completeness**: Comprehensive answers with actionable advice
- **Tool Usage**: Appropriate use of available tools
- **Reasoning**: Clear explanation of recommendations

## 🎓 Learning Objectives

This project helps students learn:

1. **Database Design**: Creating schemas for energy management systems
2. **API Integration**: Working with external weather and pricing APIs
3. **RAG Implementation**: Building retrieval-augmented generation pipelines
4. **Agent Development**: Creating intelligent agents with tool usage
5. **Evaluation Methods**: Testing and measuring agent performance
6. **Energy Optimization**: Understanding smart home energy management
7. **LLM Orchestration**: Managing multi-tool LLM workflows

## 📚 Detailed Documentation

For comprehensive setup and implementation details, see [ecohome/README.md](./ecohome/README.md)

## 🚀 Getting Started

1. Clone this repository
2. Install dependencies: `uv sync`
3. Set up environment variables in `.env`
4. Navigate to `ecohome/` directory
5. Run notebooks in sequence: `01_db_setup.ipynb` → `02_rag_setup.ipynb` → `03_run_and_evaluate.ipynb`
6. Test the agent with your own questions

## 🤝 Contributing

This is a learning project. Feel free to:
- Add new tools and capabilities
- Improve the evaluation metrics
- Enhance the RAG pipeline
- Add more sophisticated optimization algorithms
- Extend device support and use cases

## 📄 License

This project is for educational purposes as part of the Udacity Course 2 curriculum.
