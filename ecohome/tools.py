"""
Tools for EcoHome Energy Advisor Agent
"""

import os
import json
import random
import requests
from datetime import datetime, timedelta
from typing import Dict, Any
from langchain_core.tools import tool
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from models.energy import DatabaseManager

# Initialize database manager
db_manager = DatabaseManager()


@tool
def get_weather_forecast(location: str, days: int = 3) -> Dict[str, Any]:
    """
    Get a weather forecast for a given location using the OpenWeatherMap free API.

    Uses the 5-day/3-hour forecast endpoint (data/2.5/forecast), which provides
    forecasts in 3-hour intervals up to 5 days ahead. Solar irradiance is estimated
    from cloud cover and time of day since the free tier does not provide it directly.

    Args:
        location: City name or "City, Country" (e.g., "Berlin", "London, GB").
        days: Number of days to forecast, between 1 and 5 (default: 3).

    Returns:
        A dict with keys:
          - location (str): Resolved city and country.
          - forecast_days (int): Number of days of data returned.
          - current (dict): Temperature, condition, humidity, wind_speed.
          - hourly (list[dict]): 3-hour interval entries with datetime, temperature_c,
            condition, estimated_solar_irradiance_wm2, humidity, wind_speed.
          - error (str): Present only if the call failed.
    """
    # --- Guard: API key ---
    api_key = os.getenv("OPENWEATHER_API_KEY")
    if not api_key:
        return {"error": "OPENWEATHER_API_KEY is not set in environment variables."}

    # --- Clamp days to free-tier limit ---
    days = max(1, min(days, 5))

    try:
        # 1. Geocoding
        geo_url = (
            f"http://api.openweathermap.org/geo/1.0/direct"
            f"?q={location}&limit=1&appid={api_key}"
        )
        geo_res = requests.get(geo_url, timeout=10)
        geo_res.raise_for_status()
        geo_data = geo_res.json()

        if not geo_data:
            return {"error": f"Location '{location}' not found. Try 'City, CountryCode' format."}

        lat = geo_data[0]["lat"]
        lon = geo_data[0]["lon"]
        resolved_location = f"{geo_data[0]['name']}, {geo_data[0].get('country', '')}"

        # 2. 5-day / 3-hour forecast (free tier)
        forecast_url = (
            f"https://api.openweathermap.org/data/2.5/forecast"
            f"?lat={lat}&lon={lon}&units=metric&appid={api_key}"
        )
        forecast_res = requests.get(forecast_url, timeout=10)

        if forecast_res.status_code == 401:
            return {"error": "Invalid API key or key not yet active (new keys take ~60 min)."}

        forecast_res.raise_for_status()
        data = forecast_res.json()

        # 3. Current conditions — first forecast entry
        current_raw = data["list"][0]
        current = {
            "temperature_c": current_raw["main"]["temp"],
            "condition": current_raw["weather"][0]["description"],
            "humidity": current_raw["main"]["humidity"],
            "wind_speed": current_raw["wind"]["speed"],
        }

        # 4. Hourly entries (3-hour steps), limited to requested days
        limit_count = days * 8  # 8 x 3-hour slots = 24 hours per day
        hourly_list = []

        for entry in data["list"][:limit_count]:
            dt = datetime.fromtimestamp(entry["dt"])
            hour = dt.hour
            clouds = entry["clouds"]["all"]  # Cloud cover percentage (0–100)

            # Estimate solar irradiance (W/m²) from time-of-day + cloud cover.
            # Peak window: 10:00–16:00 → up to 800 W/m²
            # Shoulder window: 07:00–19:00 → up to 200 W/m²
            # Night: 0 W/m²
            if 10 <= hour <= 16:
                base_solar = 800
            elif 7 <= hour <= 19:
                base_solar = 200
            else:
                base_solar = 0

            estimated_solar = round(base_solar * (1 - clouds / 100), 2)

            hourly_list.append({
                "datetime": dt.strftime("%Y-%m-%d %H:%M"),
                "temperature_c": entry["main"]["temp"],
                "condition": entry["weather"][0]["description"],
                "estimated_solar_irradiance_wm2": estimated_solar,
                "humidity": entry["main"]["humidity"],
                "wind_speed": entry["wind"]["speed"],
            })

        # Compute actual days covered by the returned data
        if hourly_list:
            first_dt = datetime.strptime(hourly_list[0]["datetime"], "%Y-%m-%d %H:%M")
            last_dt = datetime.strptime(hourly_list[-1]["datetime"], "%Y-%m-%d %H:%M")
            actual_days = (last_dt - first_dt).days + 1
        else:
            actual_days = 0

        return {
            "location": resolved_location,
            "forecast_days": actual_days,
            "current": current,
            "hourly": hourly_list,
        }

    except requests.exceptions.Timeout:
        return {"error": "Request timed out. OpenWeatherMap may be unreachable."}
    except requests.exceptions.HTTPError as e:
        return {"error": f"HTTP error from OpenWeatherMap: {e}"}
    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}"}



# TODO: Implement get_electricity_prices tool
@tool
def get_electricity_prices(date: str = None) -> Dict[str, Any]:
    """
    Get electricity prices for a specific date or current day.

    Args:
        date (str): Date in YYYY-MM-DD format (defaults to today)

    Returns:
        Dict[str, Any]: Electricity pricing data with hourly rates
        E.g:
        prices = {
            "date": ...,
            "pricing_type": "time_of_use",
            "currency": "USD",
            "unit": "per_kWh",
            "hourly_rates": [
                {
                    "hour": .., # for hour in range(24)
                    "rate": ..,
                    "period": ..,
                    "demand_charge": ...
                }
            ]
        }
    """
    if date is None:
        date = datetime.now().strftime("%Y-%m-%d")

    # Mock electricity pricing - in real implementation, this would call a pricing API
    # Use a base price per kWh
    # Then generate hourly rates with peak/off-peak pricing
    # Peak normally between 6 and 22...
    # demand_charge should be 0 if off-peak

    return


@tool
def query_energy_usage(
    start_date: str, end_date: str, device_type: str = None
) -> Dict[str, Any]:
    """
    Query energy usage data from the database for a specific date range.

    Args:
        start_date (str): Start date in YYYY-MM-DD format
        end_date (str): End date in YYYY-MM-DD format
        device_type (str): Optional device type filter (e.g., "EV", "HVAC", "appliance")

    Returns:
        Dict[str, Any]: Energy usage data with consumption details
    """
    try:
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=1)

        records = db_manager.get_usage_by_date_range(start_dt, end_dt)

        if device_type:
            records = [r for r in records if r.device_type == device_type]

        usage_data = {
            "start_date": start_date,
            "end_date": end_date,
            "device_type": device_type,
            "total_records": len(records),
            "total_consumption_kwh": round(sum(r.consumption_kwh for r in records), 2),
            "total_cost_usd": round(sum(r.cost_usd or 0 for r in records), 2),
            "records": [],
        }

        for record in records:
            usage_data["records"].append(
                {
                    "timestamp": record.timestamp.isoformat(),
                    "consumption_kwh": record.consumption_kwh,
                    "device_type": record.device_type,
                    "device_name": record.device_name,
                    "cost_usd": record.cost_usd,
                }
            )

        return usage_data
    except Exception as e:
        return {"error": f"Failed to query energy usage: {str(e)}"}


@tool
def query_solar_generation(start_date: str, end_date: str) -> Dict[str, Any]:
    """
    Query solar generation data from the database for a specific date range.

    Args:
        start_date (str): Start date in YYYY-MM-DD format
        end_date (str): End date in YYYY-MM-DD format

    Returns:
        Dict[str, Any]: Solar generation data with production details
    """
    try:
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=1)

        records = db_manager.get_generation_by_date_range(start_dt, end_dt)

        generation_data = {
            "start_date": start_date,
            "end_date": end_date,
            "total_records": len(records),
            "total_generation_kwh": round(sum(r.generation_kwh for r in records), 2),
            "average_daily_generation": round(
                sum(r.generation_kwh for r in records)
                / max(1, (end_dt - start_dt).days),
                2,
            ),
            "records": [],
        }

        for record in records:
            generation_data["records"].append(
                {
                    "timestamp": record.timestamp.isoformat(),
                    "generation_kwh": record.generation_kwh,
                    "weather_condition": record.weather_condition,
                    "temperature_c": record.temperature_c,
                    "solar_irradiance": record.solar_irradiance,
                }
            )

        return generation_data
    except Exception as e:
        return {"error": f"Failed to query solar generation: {str(e)}"}


@tool
def get_recent_energy_summary(hours: int = 24) -> Dict[str, Any]:
    """
    Get a summary of recent energy usage and solar generation.

    Args:
        hours (int): Number of hours to look back (default 24)

    Returns:
        Dict[str, Any]: Summary of recent energy data
    """
    try:
        usage_records = db_manager.get_recent_usage(hours)
        generation_records = db_manager.get_recent_generation(hours)

        summary = {
            "time_period_hours": hours,
            "usage": {
                "total_consumption_kwh": round(
                    sum(r.consumption_kwh for r in usage_records), 2
                ),
                "total_cost_usd": round(sum(r.cost_usd or 0 for r in usage_records), 2),
                "device_breakdown": {},
            },
            "generation": {
                "total_generation_kwh": round(
                    sum(r.generation_kwh for r in generation_records), 2
                ),
                "average_weather": "sunny" if generation_records else "unknown",
            },
        }

        # Calculate device breakdown
        for record in usage_records:
            device = record.device_type or "unknown"
            if device not in summary["usage"]["device_breakdown"]:
                summary["usage"]["device_breakdown"][device] = {
                    "consumption_kwh": 0,
                    "cost_usd": 0,
                    "records": 0,
                }
            summary["usage"]["device_breakdown"][device]["consumption_kwh"] += (
                record.consumption_kwh
            )
            summary["usage"]["device_breakdown"][device]["cost_usd"] += (
                record.cost_usd or 0
            )
            summary["usage"]["device_breakdown"][device]["records"] += 1

        # Round the breakdown values
        for device_data in summary["usage"]["device_breakdown"].values():
            device_data["consumption_kwh"] = round(device_data["consumption_kwh"], 2)
            device_data["cost_usd"] = round(device_data["cost_usd"], 2)

        return summary
    except Exception as e:
        return {"error": f"Failed to get recent energy summary: {str(e)}"}


@tool
def search_energy_tips(query: str, max_results: int = 5) -> Dict[str, Any]:
    """
    Search for energy-saving tips and best practices using RAG.

    Args:
        query (str): Search query for energy tips
        max_results (int): Maximum number of results to return

    Returns:
        Dict[str, Any]: Relevant energy tips and best practices
    """
    try:
        # Initialize vector store if it doesn't exist
        persist_directory = "data/vectorstore"
        if not os.path.exists(persist_directory):
            os.makedirs(persist_directory)
            
        api_key = os.getenv("VOCAREUM_API_KEY")
        base_url = os.getenv("VOCAREUM_BASE_URL")
        print(f"Using OpenAI API key: {api_key[:4]}... and base URL: {base_url}")

        # Load documents if vector store doesn't exist
        if not os.path.exists(os.path.join(persist_directory, "chroma.sqlite3")):
            # Load documents
            documents = []
            for doc_path in [
                "data/documents/tip_device_best_practices.txt",
                "data/documents/tip_energy_savings.txt",
            ]:
                if os.path.exists(doc_path):
                    loader = TextLoader(doc_path)
                    docs = loader.load()
                    documents.extend(docs)

            # Split documents
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, chunk_overlap=200
            )
            splits = text_splitter.split_documents(documents)

            # Create vector store
            # embeddings = OpenAIEmbeddings()
            
            embeddings = OpenAIEmbeddings(
                model="text-embedding-3-small",
                base_url=base_url,
                api_key=api_key,
            )
            vectorstore = Chroma.from_documents(
                documents=splits,
                embedding=embeddings,
                persist_directory=persist_directory,
            )
        else:
            # Load existing vector store
            embeddings = OpenAIEmbeddings(
                model="text-embedding-3-small",
                base_url=base_url,
                api_key=api_key,
            )   
            
            vectorstore = Chroma(
                persist_directory=persist_directory, embedding_function=embeddings
            )

        # Search for relevant documents
        docs = vectorstore.similarity_search(query, k=max_results)

        results = {"query": query, "total_results": len(docs), "tips": []}

        for i, doc in enumerate(docs):
            results["tips"].append(
                {
                    "rank": i + 1,
                    "content": doc.page_content,
                    "source": doc.metadata.get("source", "unknown"),
                    "relevance_score": "high"
                    if i < 2
                    else "medium"
                    if i < 4
                    else "low",
                }
            )

        return results
    except Exception as e:
        return {"error": f"Failed to search energy tips: {str(e)}"}


@tool
def calculate_energy_savings(
    device_type: str,
    current_usage_kwh: float,
    optimized_usage_kwh: float,
    price_per_kwh: float = 0.12,
) -> Dict[str, Any]:
    """
    Calculate potential energy savings from optimization.

    Args:
        device_type (str): Type of device being optimized
        current_usage_kwh (float): Current energy usage in kWh
        optimized_usage_kwh (float): Optimized energy usage in kWh
        price_per_kwh (float): Price per kWh (default 0.12)

    Returns:
        Dict[str, Any]: Savings calculation results
    """
    savings_kwh = current_usage_kwh - optimized_usage_kwh
    savings_usd = savings_kwh * price_per_kwh
    savings_percentage = (
        (savings_kwh / current_usage_kwh) * 100 if current_usage_kwh > 0 else 0
    )

    return {
        "device_type": device_type,
        "current_usage_kwh": current_usage_kwh,
        "optimized_usage_kwh": optimized_usage_kwh,
        "savings_kwh": round(savings_kwh, 2),
        "savings_usd": round(savings_usd, 2),
        "savings_percentage": round(savings_percentage, 1),
        "price_per_kwh": price_per_kwh,
        "annual_savings_usd": round(savings_usd * 365, 2),
    }


TOOL_KIT = [
    get_weather_forecast,
    get_electricity_prices,
    query_energy_usage,
    query_solar_generation,
    get_recent_energy_summary,
    search_energy_tips,
    calculate_energy_savings,
]


import os
from dotenv import load_dotenv

load_dotenv() 

def test_tool_output():
    print("--- Testing LangChain Weather Tool ---")
    
    # Define a test location
    test_location = "San Francisco, US"
    
    try:
        # 2. Call the tool directly using .invoke()
        # This simulates how a LangChain Agent would trigger it
        response = get_weather_forecast.invoke({"location": test_location, "days": 3})
        
        # 3. Check for errors in the returned dictionary
        if "error" in response:
            print(f"❌ Tool Error: {response['error']}")
            return

        # 4. Print clean results
        print(f"✅ Success! Data for: {response['location']}")
        print(f"Current Temp: {response['current']['temperature_c']}°C")
        print(f"Current Condition: {response['current']['condition']}")
        
        # Show first 3 hours of forecast
        print("\n--- Forecast Snippet (First 3 Hours) ---")
        for h in response['hourly'][:3]:
            print(f"Hour {h['hour']:02d}: {h['temperature_c']}°C, {h['condition']}")

    except Exception as e:
        print(f"💥 Execution failed: {e}")

if __name__ == "__main__":
    test_tool_output()
