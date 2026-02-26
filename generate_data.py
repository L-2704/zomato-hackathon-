"""
CSAO Rail Recommendation System - Synthetic Data Generation
Step 1: Input Ingestion - Data Foundation

Generates:
  restaurants.csv  (~100 restaurants)
  menu_items.csv   (~800-1200 items)
  users.csv        (~5000 user profiles)
  order_history.csv(~30000 past orders)
  sessions.csv     (~15000 sessions)
  cart_events.csv  (sequential cart events, 8-12% rec acceptance)
"""

import os
import json
import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from faker import Faker

SEED = 42
np.random.seed(SEED)
random.seed(SEED)
fake = Faker("en_IN")
Faker.seed(SEED)

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(DATA_DIR, exist_ok=True)

# ────────────────────────────────────────────────────────────────
# CONFIGURATION
# ────────────────────────────────────────────────────────────────
NUM_RESTAURANTS = 100
NUM_USERS = 5000
NUM_ORDERS = 30000
NUM_SESSIONS = 15000

START_DATE = datetime(2025, 12, 1)
END_DATE = datetime(2026, 1, 5)  # 5 weeks

CUISINES = [
    "North Indian", "South Indian", "Chinese", "Mughlai", "Biryani",
    "Street Food", "Italian", "Continental", "Desserts & Bakery", "Cafe",
]
CUISINE_WEIGHTS = [0.25, 0.15, 0.15, 0.08, 0.10, 0.08, 0.07, 0.04, 0.04, 0.04]

CITIES = ["Mumbai", "Delhi", "Bangalore", "Hyderabad", "Chennai",
          "Kolkata", "Pune", "Ahmedabad", "Jaipur", "Lucknow"]
CITY_WEIGHTS = [0.20, 0.18, 0.15, 0.12, 0.10, 0.08, 0.07, 0.05, 0.03, 0.02]

ZONES = ["Central", "North", "South", "East", "West"]

SEGMENT_NAMES = ["Budget", "Premium", "Health", "Family", "Occasional"]
SEGMENT_WEIGHTS = [0.30, 0.15, 0.10, 0.20, 0.25]

PRICE_TIERS = ["budget", "mid", "premium"]
PRICE_TIER_WEIGHTS = [0.35, 0.45, 0.20]
PRICE_MULT = {"budget": 0.75, "mid": 1.0, "premium": 1.40}

ORDER_HOURS = [8, 9, 10, 12, 13, 14, 19, 20, 21, 22, 23, 0, 1]
HOUR_WEIGHTS = [0.03, 0.03, 0.04, 0.12, 0.12, 0.08, 0.12, 0.15, 0.12, 0.08, 0.05, 0.03, 0.03]

# ────────────────────────────────────────────────────────────────
# FOOD CATALOG
# Regular item tuple: (name, veg, price_lo, price_hi, subcat, desc)
# Combo  item tuple : (name, veg, price_lo, price_hi, subcat, desc, [components])
# ────────────────────────────────────────────────────────────────

FOOD_CATALOG = {
    # ── North Indian ──────────────────────────────────────────
    "North Indian": {
        "main": [
            ("Butter Chicken", False, 249, 349, "curry", "Tender chicken in creamy tomato gravy"),
            ("Paneer Butter Masala", True, 209, 299, "curry", "Cottage cheese in rich tomato gravy"),
            ("Dal Makhani", True, 179, 259, "curry", "Slow-cooked black lentils"),
            ("Shahi Paneer", True, 219, 309, "curry", "Paneer in creamy cashew gravy"),
            ("Kadai Chicken", False, 249, 349, "curry", "Chicken with capsicum and spices"),
            ("Kadai Paneer", True, 199, 289, "curry", "Paneer with capsicum and onion"),
            ("Palak Paneer", True, 189, 269, "curry", "Paneer in spinach gravy"),
            ("Chole Masala", True, 149, 219, "curry", "Spiced chickpea curry"),
            ("Rajma Masala", True, 149, 219, "curry", "Kidney beans in thick gravy"),
            ("Mixed Veg Curry", True, 159, 229, "curry", "Seasonal vegetables in spiced gravy"),
            ("Malai Kofta", True, 219, 309, "curry", "Paneer balls in creamy gravy"),
            ("Chicken Tikka Masala", False, 259, 369, "curry", "Grilled chicken in spiced masala"),
            ("Mutton Rogan Josh", False, 329, 449, "curry", "Slow-cooked mutton Kashmiri style"),
            ("Egg Curry", False, 149, 219, "curry", "Boiled eggs in onion-tomato gravy"),
        ],
        "bread": [
            ("Butter Naan", True, 45, 69, "naan", "Soft leavened bread with butter"),
            ("Garlic Naan", True, 55, 79, "naan", "Naan with garlic and coriander"),
            ("Tandoori Roti", True, 30, 49, "roti", "Whole wheat bread from tandoor"),
            ("Butter Roti", True, 35, 55, "roti", "Soft roti brushed with butter"),
            ("Laccha Paratha", True, 50, 79, "paratha", "Layered flaky flatbread"),
            ("Aloo Paratha", True, 59, 89, "paratha", "Stuffed potato flatbread"),
            ("Stuffed Kulcha", True, 69, 99, "naan", "Naan stuffed with paneer or onion"),
            ("Missi Roti", True, 40, 59, "roti", "Gram flour spiced flatbread"),
        ],
        "rice": [
            ("Jeera Rice", True, 119, 179, "rice", "Cumin-tempered basmati rice"),
            ("Steamed Rice", True, 99, 149, "rice", "Plain steamed basmati rice"),
            ("Veg Pulao", True, 149, 219, "rice", "Rice with mixed vegetables"),
        ],
        "side": [
            ("Boondi Raita", True, 49, 79, "accompaniment", "Yogurt with crispy boondi"),
            ("Roasted Papad", True, 25, 39, "accompaniment", "Crispy roasted lentil wafer"),
            ("Green Salad", True, 39, 69, "salad", "Cucumber, tomato, onion salad"),
            ("Onion Rings", True, 29, 49, "accompaniment", "Sliced onions with lemon"),
            ("Mixed Pickle", True, 25, 39, "accompaniment", "Spiced mango and lime pickle"),
        ],
        "beverage": [
            ("Sweet Lassi", True, 59, 99, "lassi", "Chilled sweetened yogurt drink"),
            ("Mango Lassi", True, 79, 119, "lassi", "Creamy mango yogurt shake"),
            ("Masala Chaas", True, 39, 59, "buttermilk", "Spiced buttermilk with cumin"),
            ("Coca Cola 300ml", True, 40, 60, "soft_drink", "Chilled cola"),
            ("Thumbs Up 300ml", True, 40, 60, "soft_drink", "Chilled Thumbs Up"),
            ("Water Bottle 1L", True, 20, 30, "water", "Packaged drinking water"),
        ],
        "dessert": [
            ("Gulab Jamun (2 Pcs)", True, 59, 99, "indian_sweet", "Milk balls in sugar syrup"),
            ("Rasmalai (2 Pcs)", True, 79, 129, "indian_sweet", "Paneer discs in sweet milk"),
            ("Kheer", True, 69, 109, "indian_sweet", "Creamy rice pudding with nuts"),
            ("Gajar Ka Halwa", True, 79, 119, "indian_sweet", "Warm carrot pudding with ghee"),
        ],
        "appetizer": [
            ("Paneer Tikka (6 Pcs)", True, 189, 269, "tikka", "Grilled marinated paneer cubes"),
            ("Chicken Tikka (6 Pcs)", False, 209, 309, "tikka", "Grilled marinated chicken"),
            ("Tandoori Chicken Half", False, 229, 339, "tandoori", "Half chicken roasted in tandoor"),
            ("Hara Bhara Kebab (4 Pcs)", True, 159, 239, "kebab", "Spinach and pea patties"),
            ("Seekh Kebab (4 Pcs)", False, 199, 299, "kebab", "Minced mutton skewers"),
        ],
        "combo": [
            ("North Indian Thali Veg", True, 249, 349, "thali",
             "Dal, paneer curry, rice, 2 roti, salad, sweet",
             ["curry", "curry", "rice", "roti", "roti", "salad", "indian_sweet"]),
            ("North Indian Thali Non-Veg", False, 299, 399, "thali",
             "Chicken curry, dal, rice, 2 naan, raita, sweet",
             ["curry", "curry", "rice", "naan", "naan", "accompaniment", "indian_sweet"]),
            ("Butter Chicken Meal", False, 329, 429, "meal_combo",
             "Butter chicken, 2 naan, jeera rice, raita",
             ["curry", "naan", "naan", "rice", "accompaniment"]),
        ],
    },
    # ── South Indian ──────────────────────────────────────────
    "South Indian": {
        "main": [
            ("Plain Dosa", True, 79, 119, "dosa", "Crispy rice and lentil crepe"),
            ("Masala Dosa", True, 99, 149, "dosa", "Dosa with spiced potato filling"),
            ("Rava Dosa", True, 99, 139, "dosa", "Crispy semolina crepe"),
            ("Mysore Masala Dosa", True, 109, 159, "dosa", "Dosa with red chutney and potato"),
            ("Idli (3 Pcs)", True, 59, 89, "idli", "Steamed rice cakes"),
            ("Medu Vada (2 Pcs)", True, 69, 99, "vada", "Crispy urad dal fritters"),
            ("Uttapam", True, 89, 129, "uttapam", "Thick pancake with onion and tomato"),
            ("Set Dosa (3 Pcs)", True, 89, 129, "dosa", "Soft spongy mini dosas"),
            ("Pongal", True, 79, 119, "rice_dish", "Spiced rice and lentil porridge"),
            ("Chicken Chettinad", False, 229, 329, "curry", "Spicy chicken Chettinad masala"),
        ],
        "side": [
            ("Sambar", True, 39, 59, "accompaniment", "Lentil and vegetable stew"),
            ("Coconut Chutney", True, 29, 49, "accompaniment", "Fresh ground coconut chutney"),
            ("Tomato Chutney", True, 29, 49, "accompaniment", "Tangy tomato chutney"),
        ],
        "rice": [
            ("Curd Rice", True, 89, 129, "rice", "Yogurt rice with tempering"),
            ("Lemon Rice", True, 99, 139, "rice", "Tangy lemon-flavoured rice"),
            ("Bisi Bele Bath", True, 119, 169, "rice", "Spiced rice with lentils"),
        ],
        "beverage": [
            ("Filter Coffee", True, 39, 69, "coffee", "Traditional South Indian coffee"),
            ("Buttermilk", True, 29, 49, "buttermilk", "Spiced churned yogurt drink"),
            ("Rose Milk", True, 49, 79, "milk_drink", "Chilled rose-flavoured milk"),
        ],
        "dessert": [
            ("Payasam", True, 69, 109, "indian_sweet", "Vermicelli kheer with nuts"),
            ("Mysore Pak", True, 49, 79, "indian_sweet", "Rich gram flour fudge"),
            ("Kesari Bath", True, 49, 79, "indian_sweet", "Semolina sweet with saffron"),
        ],
        "combo": [
            ("South Indian Thali", True, 199, 279, "thali",
             "Rice, sambar, rasam, 2 curries, papad, payasam",
             ["rice", "accompaniment", "curry", "curry", "accompaniment", "indian_sweet"]),
            ("Mini Tiffin Combo", True, 149, 199, "meal_combo",
             "2 Idli, 1 vada, 1 dosa, sambar, chutney",
             ["idli", "vada", "dosa", "accompaniment", "accompaniment"]),
        ],
    },
    # ── Chinese ───────────────────────────────────────────────
    "Chinese": {
        "main": [
            ("Veg Fried Rice", True, 149, 219, "fried_rice", "Wok-tossed rice with vegetables"),
            ("Chicken Fried Rice", False, 179, 259, "fried_rice", "Rice tossed with chicken"),
            ("Egg Fried Rice", False, 159, 229, "fried_rice", "Rice with scrambled egg"),
            ("Veg Hakka Noodles", True, 149, 219, "noodles", "Stir-fried noodles with veggies"),
            ("Chicken Hakka Noodles", False, 179, 259, "noodles", "Noodles tossed with chicken"),
            ("Schezwan Noodles", True, 159, 229, "noodles", "Spicy Schezwan-style noodles"),
            ("Veg Manchurian Gravy", True, 159, 229, "manchurian", "Veggie balls in Chinese gravy"),
            ("Chicken Manchurian", False, 189, 269, "manchurian", "Chicken in Manchurian sauce"),
            ("Chilli Chicken", False, 199, 279, "dry_prep", "Chicken tossed in chilli sauce"),
            ("Chilli Paneer", True, 179, 259, "dry_prep", "Paneer in spicy chilli sauce"),
            ("Kung Pao Chicken", False, 219, 299, "dry_prep", "Chicken with peanuts"),
            ("Dragon Chicken", False, 219, 299, "dry_prep", "Crispy chicken in dragon sauce"),
        ],
        "side": [
            ("Veg Spring Rolls (4 Pcs)", True, 109, 159, "spring_roll", "Crispy veggie rolls"),
            ("Chicken Spring Rolls (4 Pcs)", False, 129, 179, "spring_roll", "Crispy chicken rolls"),
            ("Veg Dim Sum (6 Pcs)", True, 129, 189, "dimsum", "Steamed veggie dumplings"),
            ("Chicken Momos (6 Pcs)", False, 119, 179, "momos", "Steamed chicken dumplings"),
            ("Veg Momos (6 Pcs)", True, 99, 159, "momos", "Steamed vegetable dumplings"),
        ],
        "soup": [
            ("Hot and Sour Soup", True, 99, 149, "soup", "Spicy and tangy soup"),
            ("Manchow Soup", True, 99, 149, "soup", "Thick soup with fried noodles"),
            ("Sweet Corn Soup", True, 89, 139, "soup", "Creamy sweet corn soup"),
            ("Chicken Sweet Corn Soup", False, 109, 159, "soup", "Corn soup with chicken"),
        ],
        "beverage": [
            ("Iced Lemon Tea", True, 59, 89, "iced_tea", "Chilled lemon tea"),
            ("Coca Cola 300ml", True, 40, 60, "soft_drink", "Chilled cola"),
            ("Fresh Lime Soda", True, 49, 79, "soda", "Fresh lime with soda water"),
        ],
        "dessert": [
            ("Honey Noodles with Ice Cream", True, 129, 179, "western_dessert", "Crispy noodles with honey and ice cream"),
            ("Date Pancake", True, 99, 149, "western_dessert", "Sweet date-filled pancakes"),
        ],
        "combo": [
            ("Chinese Combo for 1", True, 199, 279, "meal_combo",
             "Fried rice, noodles, manchurian, soup",
             ["fried_rice", "noodles", "manchurian", "soup"]),
            ("Chinese Feast for 2", True, 399, 499, "meal_combo",
             "2 Rice, noodles, manchurian, spring rolls, soup",
             ["fried_rice", "fried_rice", "noodles", "manchurian", "spring_roll", "soup"]),
        ],
    },
    # ── Mughlai ───────────────────────────────────────────────
    "Mughlai": {
        "main": [
            ("Murgh Musallam", False, 349, 449, "curry", "Whole spiced chicken in rich gravy"),
            ("Nihari", False, 299, 399, "curry", "Slow-cooked meat stew"),
            ("Chicken Korma", False, 269, 369, "curry", "Mild creamy chicken with nuts"),
            ("Veg Korma", True, 219, 299, "curry", "Vegetables in cashew gravy"),
            ("Mutton Do Pyaza", False, 319, 419, "curry", "Mutton with double onions"),
            ("Keema Matar", False, 229, 309, "curry", "Minced meat with green peas"),
            ("Mughlai Chicken", False, 279, 369, "curry", "Chicken in Mughlai spices"),
        ],
        "bread": [
            ("Sheermal", True, 59, 89, "naan", "Sweet saffron bread"),
            ("Roomali Roti", True, 39, 59, "roti", "Paper-thin handkerchief bread"),
            ("Naan", True, 39, 59, "naan", "Plain leavened oven bread"),
            ("Garlic Naan", True, 55, 79, "naan", "Naan with garlic topping"),
            ("Tandoori Paratha", True, 49, 69, "paratha", "Layered bread from tandoor"),
        ],
        "rice": [
            ("Mughlai Pulao", True, 169, 249, "rice", "Rice with dry fruits and saffron"),
            ("Zafrani Rice", True, 149, 219, "rice", "Saffron-infused basmati rice"),
            ("Jeera Rice", True, 119, 169, "rice", "Cumin-tempered rice"),
        ],
        "side": [
            ("Mint Raita", True, 49, 79, "accompaniment", "Yogurt with fresh mint"),
            ("Sirka Pyaz", True, 29, 49, "accompaniment", "Vinegar-marinated onions"),
            ("Green Salad", True, 39, 59, "salad", "Fresh seasonal salad"),
        ],
        "appetizer": [
            ("Galouti Kebab (4 Pcs)", False, 249, 349, "kebab", "Melt-in-mouth mutton kebabs"),
            ("Shami Kebab (4 Pcs)", False, 199, 289, "kebab", "Spiced minced meat patties"),
            ("Tandoori Paneer Tikka", True, 189, 269, "tikka", "Grilled marinated paneer"),
            ("Mughlai Chicken Wings", False, 199, 279, "tandoori", "Spiced chicken wings"),
        ],
        "beverage": [
            ("Thandai", True, 79, 119, "milk_drink", "Spiced cold milk with nuts"),
            ("Rose Sharbat", True, 49, 79, "sharbat", "Sweet rose-flavoured drink"),
            ("Coca Cola 300ml", True, 40, 60, "soft_drink", "Chilled cola"),
        ],
        "dessert": [
            ("Phirni", True, 79, 119, "indian_sweet", "Creamy ground rice pudding"),
            ("Shahi Tukda", True, 89, 129, "indian_sweet", "Fried bread soaked in rabri"),
            ("Kulfi", True, 69, 99, "indian_sweet", "Traditional Indian ice cream"),
        ],
        "combo": [
            ("Mughlai Feast", False, 399, 549, "thali",
             "Nihari, kebab, biryani, naan, phirni",
             ["curry", "kebab", "biryani", "naan", "indian_sweet"]),
            ("Royal Mughlai Thali", False, 449, 599, "thali",
             "Korma, seekh kebab, pulao, roomali, kulfi",
             ["curry", "kebab", "rice", "roti", "indian_sweet"]),
        ],
    },
    # ── Biryani ───────────────────────────────────────────────
    "Biryani": {
        "main": [
            ("Chicken Biryani", False, 219, 319, "biryani", "Layered rice with spiced chicken"),
            ("Mutton Biryani", False, 279, 399, "biryani", "Layered rice with tender mutton"),
            ("Veg Biryani", True, 179, 259, "biryani", "Fragrant rice with mixed vegetables"),
            ("Egg Biryani", False, 169, 249, "biryani", "Biryani with boiled eggs"),
            ("Hyderabadi Dum Biryani", False, 269, 379, "biryani", "Slow-cooked Hyderabadi biryani"),
            ("Lucknowi Biryani", False, 249, 349, "biryani", "Aromatic Awadhi-style biryani"),
            ("Paneer Biryani", True, 199, 279, "biryani", "Biryani with spiced paneer"),
            ("Prawn Biryani", False, 299, 419, "biryani", "Rice layered with spiced prawns"),
        ],
        "side": [
            ("Mirchi Ka Salan", True, 69, 109, "salan", "Green chilli curry Hyderabadi"),
            ("Raita", True, 49, 79, "accompaniment", "Onion and tomato yogurt"),
            ("Salan", True, 59, 89, "salan", "Peanut and sesame seed gravy"),
            ("Boiled Egg (2 Pcs)", False, 39, 59, "accompaniment", "Boiled eggs"),
        ],
        "beverage": [
            ("Pepsi 300ml", True, 40, 60, "soft_drink", "Chilled Pepsi"),
            ("Sweet Lassi", True, 59, 99, "lassi", "Thick sweet yogurt drink"),
            ("Coca Cola 300ml", True, 40, 60, "soft_drink", "Chilled cola"),
            ("Water Bottle 1L", True, 20, 30, "water", "Packaged drinking water"),
        ],
        "dessert": [
            ("Double Ka Meetha", True, 79, 119, "indian_sweet", "Bread pudding with rabri"),
            ("Qubani Ka Meetha", True, 89, 129, "indian_sweet", "Apricot dessert with cream"),
        ],
        "appetizer": [
            ("Chicken 65", False, 179, 259, "dry_prep", "Spicy deep-fried chicken"),
            ("Paneer 65", True, 159, 239, "dry_prep", "Spicy deep-fried paneer"),
            ("Apollo Fish", False, 199, 279, "dry_prep", "Crispy spiced fish fry"),
        ],
        "combo": [
            ("Biryani Combo", False, 279, 369, "meal_combo",
             "Chicken biryani, raita, salan, drink",
             ["biryani", "accompaniment", "salan", "soft_drink"]),
            ("Family Biryani Pack", False, 499, 699, "meal_combo",
             "Full biryani, 4 eggs, raita, salan, 2 drinks",
             ["biryani", "accompaniment", "accompaniment", "salan", "soft_drink", "soft_drink"]),
        ],
    },
    # ── Street Food ───────────────────────────────────────────
    "Street Food": {
        "main": [
            ("Pav Bhaji", True, 89, 139, "pav_bhaji", "Spiced mashed veg with buttered pav"),
            ("Vada Pav", True, 29, 49, "vada_pav", "Potato fritter in bread bun"),
            ("Pani Puri (6 Pcs)", True, 49, 79, "chaat", "Crispy puris with spiced water"),
            ("Bhel Puri", True, 49, 79, "chaat", "Puffed rice with chutneys"),
            ("Sev Puri (6 Pcs)", True, 59, 89, "chaat", "Puris topped with sev and chutney"),
            ("Dahi Puri (6 Pcs)", True, 69, 99, "chaat", "Puris with yogurt and chutney"),
            ("Aloo Tikki (2 Pcs)", True, 49, 79, "tikki", "Crispy potato patties"),
            ("Chole Bhature", True, 99, 149, "bhature", "Chickpeas with fried bread"),
            ("Samosa (2 Pcs)", True, 39, 59, "samosa", "Crispy potato-filled pastry"),
            ("Kathi Roll Paneer", True, 99, 149, "roll", "Paneer wrapped in paratha"),
        ],
        "side": [
            ("Extra Pav (2 Pcs)", True, 20, 30, "bread", "Buttered pav bread"),
            ("Green Chutney", True, 15, 25, "accompaniment", "Mint and coriander chutney"),
            ("Tamarind Chutney", True, 15, 25, "accompaniment", "Sweet tangy tamarind sauce"),
        ],
        "beverage": [
            ("Masala Chai", True, 20, 39, "tea", "Spiced Indian tea"),
            ("Fresh Lime Soda", True, 39, 69, "soda", "Fresh lime with soda"),
            ("Sugarcane Juice", True, 39, 59, "juice", "Fresh pressed sugarcane"),
            ("Coca Cola 300ml", True, 40, 60, "soft_drink", "Chilled cola"),
            ("Mango Shake", True, 69, 99, "shake", "Fresh mango milkshake"),
        ],
        "dessert": [
            ("Rabri Falooda", True, 89, 129, "falooda", "Vermicelli with rabri and ice cream"),
            ("Kulfi Stick", True, 39, 59, "indian_sweet", "Traditional ice cream bar"),
            ("Jalebi (250g)", True, 59, 89, "indian_sweet", "Crispy syrup-soaked spirals"),
        ],
        "combo": [
            ("Street Food Platter", True, 149, 199, "meal_combo",
             "Pav bhaji, samosa, bhel puri, chai",
             ["pav_bhaji", "samosa", "chaat", "tea"]),
            ("Chaat Festival", True, 179, 249, "meal_combo",
             "Pani puri, sev puri, dahi puri, tikki, soda",
             ["chaat", "chaat", "chaat", "tikki", "soda"]),
        ],
    },
    # ── Italian ───────────────────────────────────────────────
    "Italian": {
        "main": [
            ("Margherita Pizza Regular", True, 199, 279, "pizza", "Classic tomato and mozzarella"),
            ("Pepperoni Pizza Regular", False, 249, 349, "pizza", "Pizza with spicy pepperoni"),
            ("Farmhouse Pizza Regular", True, 229, 319, "pizza", "Pizza with veggies and mushrooms"),
            ("Penne Arrabiata", True, 179, 259, "pasta", "Penne in spicy tomato sauce"),
            ("Spaghetti Bolognese", False, 219, 299, "pasta", "Spaghetti with meat sauce"),
            ("Alfredo Pasta", True, 199, 279, "pasta", "Creamy white sauce pasta"),
            ("Mushroom Risotto", True, 219, 299, "risotto", "Creamy rice with mushrooms"),
            ("Lasagna", False, 249, 349, "pasta", "Layered pasta with meat and cheese"),
        ],
        "side": [
            ("Garlic Bread (4 Pcs)", True, 89, 129, "bread", "Toasted garlic butter bread"),
            ("Cheesy Garlic Bread", True, 109, 159, "bread", "Garlic bread with cheese"),
            ("Caesar Salad", True, 129, 179, "salad", "Romaine with Caesar dressing"),
            ("Bruschetta (4 Pcs)", True, 109, 149, "appetizer", "Toasted bread with tomato basil"),
        ],
        "beverage": [
            ("Iced Tea Peach", True, 69, 99, "iced_tea", "Chilled peach iced tea"),
            ("Virgin Mojito", True, 89, 129, "mocktail", "Mint and lime refresher"),
            ("Coca Cola 300ml", True, 40, 60, "soft_drink", "Chilled cola"),
            ("Cold Coffee", True, 79, 119, "coffee", "Blended cold coffee"),
        ],
        "dessert": [
            ("Tiramisu", True, 149, 199, "western_dessert", "Coffee-flavoured Italian dessert"),
            ("Chocolate Brownie", True, 99, 149, "western_dessert", "Warm chocolate brownie"),
            ("Panna Cotta", True, 129, 179, "western_dessert", "Italian cream dessert"),
        ],
        "combo": [
            ("Italian Meal for 1", True, 299, 399, "meal_combo",
             "Pizza, pasta, garlic bread, drink",
             ["pizza", "pasta", "bread", "soft_drink"]),
            ("Pizza Party for 2", True, 499, 649, "meal_combo",
             "2 Pizzas, garlic bread, 2 drinks, brownie",
             ["pizza", "pizza", "bread", "soft_drink", "soft_drink", "western_dessert"]),
        ],
    },
    # ── Continental ───────────────────────────────────────────
    "Continental": {
        "main": [
            ("Grilled Chicken Steak", False, 279, 379, "steak", "Herb-marinated grilled chicken"),
            ("Fish and Chips", False, 249, 349, "fish", "Battered fish with crispy fries"),
            ("Veg Burger", True, 129, 179, "burger", "Veggie patty burger with fries"),
            ("Chicken Burger", False, 149, 219, "burger", "Grilled chicken burger"),
            ("Club Sandwich Veg", True, 139, 199, "sandwich", "Triple-decker veggie sandwich"),
            ("Club Sandwich Chicken", False, 159, 229, "sandwich", "Triple-decker chicken sandwich"),
        ],
        "side": [
            ("French Fries", True, 79, 119, "fries", "Crispy golden potato fries"),
            ("Coleslaw", True, 49, 79, "salad", "Creamy cabbage salad"),
            ("Onion Rings (8 Pcs)", True, 89, 129, "appetizer", "Batter-fried onion rings"),
        ],
        "beverage": [
            ("Fresh Orange Juice", True, 79, 119, "juice", "Freshly squeezed orange juice"),
            ("Cold Coffee", True, 79, 109, "coffee", "Blended iced coffee"),
            ("Coca Cola 300ml", True, 40, 60, "soft_drink", "Chilled cola"),
        ],
        "dessert": [
            ("Chocolate Lava Cake", True, 149, 199, "western_dessert", "Cake with molten chocolate center"),
            ("New York Cheesecake", True, 169, 229, "western_dessert", "Classic baked cheesecake"),
            ("Ice Cream Sundae", True, 109, 159, "western_dessert", "Ice cream with toppings"),
        ],
        "appetizer": [
            ("Soup of the Day", True, 89, 129, "soup", "Daily soup selection"),
            ("Nachos with Salsa", True, 129, 179, "appetizer", "Corn chips with salsa and cheese"),
            ("Chicken Wings (6 Pcs)", False, 179, 249, "wings", "Crispy wings with dip"),
        ],
    },
    # ── Desserts & Bakery ─────────────────────────────────────
    "Desserts & Bakery": {
        "dessert": [
            ("Chocolate Truffle Cake Slice", True, 99, 149, "cake", "Rich chocolate layered cake"),
            ("Red Velvet Cake Slice", True, 109, 159, "cake", "Cream cheese red velvet"),
            ("Blueberry Cheesecake Slice", True, 129, 179, "cake", "Baked blueberry cheesecake"),
            ("Chocolate Mousse", True, 89, 129, "mousse", "Light airy chocolate mousse"),
            ("Brownie with Ice Cream", True, 109, 159, "brownie", "Brownie with vanilla ice cream"),
            ("Gulab Jamun (4 Pcs)", True, 79, 119, "indian_sweet", "Milk balls in sugar syrup"),
            ("Rasgulla (4 Pcs)", True, 69, 99, "indian_sweet", "Paneer balls in sugar syrup"),
            ("Butterscotch Pastry", True, 49, 79, "pastry", "Butterscotch cream pastry"),
            ("Veg Puff", True, 29, 49, "puff", "Flaky pastry with veggie filling"),
            ("Chicken Puff", False, 39, 59, "puff", "Flaky pastry with chicken filling"),
            ("Cookie Box (6 Pcs)", True, 99, 149, "cookies", "Assorted freshly baked cookies"),
            ("Fruit Tart", True, 89, 129, "tart", "Butter tart with custard and fruits"),
        ],
        "beverage": [
            ("Hot Chocolate", True, 89, 129, "hot_drink", "Rich hot chocolate with cream"),
            ("Cappuccino", True, 79, 109, "coffee", "Classic Italian cappuccino"),
            ("Cold Coffee", True, 79, 109, "coffee", "Blended iced coffee"),
            ("Fresh Juice", True, 69, 99, "juice", "Seasonal fresh fruit juice"),
        ],
        "combo": [
            ("Dessert Box", True, 249, 349, "meal_combo",
             "4 Assorted mini desserts",
             ["cake", "mousse", "brownie", "pastry"]),
            ("Cake and Coffee Combo", True, 149, 199, "meal_combo",
             "Cake slice with cappuccino",
             ["cake", "coffee"]),
        ],
    },
    # ── Cafe ──────────────────────────────────────────────────
    "Cafe": {
        "beverage": [
            ("Espresso", True, 69, 99, "coffee", "Strong Italian espresso"),
            ("Americano", True, 79, 109, "coffee", "Espresso with hot water"),
            ("Latte", True, 99, 139, "coffee", "Espresso with steamed milk"),
            ("Cappuccino", True, 89, 119, "coffee", "Espresso with foamed milk"),
            ("Cold Brew", True, 109, 149, "coffee", "Slow-steeped cold coffee"),
            ("Matcha Latte", True, 129, 169, "tea", "Japanese green tea latte"),
            ("Iced Caramel Latte", True, 129, 169, "coffee", "Cold latte with caramel"),
            ("Fresh Orange Juice", True, 89, 119, "juice", "Freshly squeezed OJ"),
            ("Berry Smoothie", True, 119, 159, "smoothie", "Blended mixed berry smoothie"),
            ("Masala Chai", True, 49, 69, "tea", "Spiced Indian tea"),
        ],
        "side": [
            ("Grilled Sandwich Veg", True, 99, 139, "sandwich", "Toasted veggie sandwich"),
            ("Grilled Sandwich Chicken", False, 119, 159, "sandwich", "Toasted chicken sandwich"),
            ("Croissant", True, 59, 89, "pastry", "Buttery flaky French pastry"),
            ("Bagel with Cream Cheese", True, 79, 109, "bread", "Toasted bagel with cream cheese"),
            ("Waffle", True, 109, 149, "waffle", "Belgian waffle with maple syrup"),
        ],
        "dessert": [
            ("Chocolate Chip Muffin", True, 59, 89, "muffin", "Freshly baked choc chip muffin"),
            ("Blueberry Muffin", True, 59, 89, "muffin", "Freshly baked blueberry muffin"),
            ("Banana Bread", True, 69, 99, "bread", "Moist banana bread slice"),
            ("Cookie", True, 39, 59, "cookies", "Large freshly baked cookie"),
        ],
        "combo": [
            ("Breakfast Combo", True, 149, 199, "meal_combo",
             "Croissant, eggs, coffee",
             ["pastry", "bread", "coffee"]),
            ("Coffee and Cake", True, 139, 179, "meal_combo",
             "Any coffee with cake slice",
             ["coffee", "cake"]),
        ],
    },
}

RESTAURANT_NAME_TEMPLATES = {
    "North Indian":      ["{}'s Dhaba", "Punjab {}", "{} Kitchen", "Desi {}", "{} Da Dhaba"],
    "South Indian":      ["{} Tiffins", "Sagar {}", "Udupi {}", "{} Dosa Corner", "Madras {}"],
    "Chinese":           ["{} Wok", "Dragon {}", "China {}", "{} Chinese Kitchen", "Orient {}"],
    "Mughlai":           ["{} Darbar", "Royal {}", "Nawab {}", "{} Mughlai House"],
    "Biryani":           ["{} Biryani House", "Paradise {}", "{} Biryani Centre", "Royal {} Biryani"],
    "Street Food":       ["{}'s Chaat", "Bombay {}", "{} Street Bites", "Chaat {}", "Dilli {}"],
    "Italian":           ["{}'s Pizzeria", "La {}", "Pizza {}", "{} Italian Kitchen"],
    "Continental":       ["{}'s Cafe", "The {} Kitchen", "{} Diner", "Urban {}"],
    "Desserts & Bakery": ["{} Bakery", "Sweet {}", "{} Cakes", "The {} Patisserie"],
    "Cafe":              ["Cafe {}", "{}'s Coffee", "The {} Bean", "{} Brew"],
}


# ────────────────────────────────────────────────────────────────
# HELPERS
# ────────────────────────────────────────────────────────────────

def meal_period(hour: int) -> str:
    if 7 <= hour < 11:
        return "breakfast"
    if 11 <= hour < 15:
        return "lunch"
    if 19 <= hour < 23:
        return "dinner"
    return "late_night"


def random_timestamp() -> datetime:
    days = np.random.uniform(0, (END_DATE - START_DATE).days)
    hour = int(np.random.choice(ORDER_HOURS, p=HOUR_WEIGHTS))
    minute = np.random.randint(0, 60)
    return START_DATE + timedelta(days=days, hours=hour, minutes=minute)


# ────────────────────────────────────────────────────────────────
# GENERATORS
# ────────────────────────────────────────────────────────────────

def generate_restaurants(n: int = NUM_RESTAURANTS) -> pd.DataFrame:
    rows = []
    used_names: set[str] = set()
    for i in range(n):
        cuisine = np.random.choice(CUISINES, p=CUISINE_WEIGHTS)
        city = np.random.choice(CITIES, p=CITY_WEIGHTS)
        zone = np.random.choice(ZONES)
        tier = np.random.choice(PRICE_TIERS, p=PRICE_TIER_WEIGHTS)
        rating = round(float(np.clip(np.random.normal(3.8, 0.5), 2.5, 5.0)), 1)
        avg_prep = int(np.clip(np.random.normal(25, 8), 10, 50))

        free_del = int(np.random.choice([0, 149, 199, 249, 299],
                                        p=[0.20, 0.20, 0.30, 0.20, 0.10]))
        thresholds = []
        if np.random.random() < 0.60:
            thresholds.append({"type": "percentage",
                               "min_order": int(np.random.choice([299, 399, 499])),
                               "discount_pct": int(np.random.choice([10, 15, 20]))})
        if np.random.random() < 0.30:
            thresholds.append({"type": "free_item",
                               "min_order": int(np.random.choice([399, 499, 599])),
                               "free_item": "Free Dessert"})

        templates = RESTAURANT_NAME_TEMPLATES.get(cuisine, ["{} Restaurant"])
        for _ in range(20):
            name = np.random.choice(templates).format(fake.last_name())
            if name not in used_names:
                break
        used_names.add(name)

        rows.append({
            "restaurant_id": f"R{i+1:04d}",
            "name": name,
            "city": city,
            "zone": zone,
            "primary_cuisine": cuisine,
            "cuisine_tags": json.dumps([cuisine]),
            "price_tier": tier,
            "rating": rating,
            "avg_prep_time": avg_prep,
            "free_delivery_min": free_del,
            "discount_thresholds": json.dumps(thresholds),
        })
    return pd.DataFrame(rows)


def generate_menu_items(restaurants: pd.DataFrame) -> pd.DataFrame:
    rows = []
    iid = 1
    for _, rest in restaurants.iterrows():
        cuisine = rest["primary_cuisine"]
        pmult = PRICE_MULT[rest["price_tier"]]
        catalog = FOOD_CATALOG.get(cuisine, {})

        for category, items in catalog.items():
            if category == "combo":
                keep = min(len(items), max(1, np.random.randint(1, len(items) + 1)))
            else:
                keep = max(1, int(len(items) * np.random.uniform(0.60, 0.95)))
            selected = random.sample(items, min(keep, len(items)))

            for tup in selected:
                is_combo = len(tup) == 7
                name, veg, plo, phi, subcat, desc = tup[:6]
                components = list(tup[6]) if is_combo else []

                price = int(np.random.uniform(plo, phi) * pmult)
                margin = round(float(np.clip(np.random.normal(35, 15), 5, 65)), 1)
                avail = bool(np.random.random() < 0.95)
                best = bool(np.random.random() < 0.15)
                prep = int(np.clip(np.random.normal(20, 8), 5, 45))
                pop = round(float(np.random.uniform(0.1, 1.0)), 2)
                pop_meal = {
                    "breakfast": round(float(np.random.uniform(0.0, 0.4)), 2),
                    "lunch":    round(float(np.random.uniform(0.3, 1.0)), 2),
                    "dinner":   round(float(np.random.uniform(0.3, 1.0)), 2),
                    "late_night": round(float(np.random.uniform(0.1, 0.6)), 2),
                }

                rows.append({
                    "item_id": f"I{iid:05d}",
                    "restaurant_id": rest["restaurant_id"],
                    "name": name,
                    "description": desc,
                    "price": price,
                    "category": "combo" if is_combo else category,
                    "subcategory": subcat,
                    "cuisine_tag": cuisine,
                    "veg_flag": veg,
                    "is_combo": is_combo,
                    "combo_components": json.dumps(components),
                    "bestseller_flag": best,
                    "availability": avail,
                    "margin_pct": margin,
                    "prep_time_mins": prep,
                    "popularity_score": pop,
                    "popularity_by_meal": json.dumps(pop_meal),
                })
                iid += 1
    return pd.DataFrame(rows)


def generate_users(n: int = NUM_USERS) -> pd.DataFrame:
    rows = []
    for i in range(n):
        seg = np.random.choice(SEGMENT_NAMES, p=SEGMENT_WEIGHTS)
        city = np.random.choice(CITIES, p=CITY_WEIGHTS)

        if seg == "Health":
            diet = np.random.choice(["veg", "vegan", "non-veg", "none"],
                                    p=[0.40, 0.20, 0.20, 0.20])
        else:
            diet = np.random.choice(["veg", "non-veg", "none"],
                                    p=[0.35, 0.45, 0.20])

        n_vd = int(np.random.choice([0, 1, 2, 3], p=[0.50, 0.25, 0.15, 0.10]))
        veg_days = sorted(random.sample(range(7), n_vd)) if n_vd else []

        rfm_cfg = {
            "Budget":     {"r": (1, 10),  "f": (10, 30), "m": (150, 350),
                           "aov": (150, 300),  "oc": (15, 50)},
            "Premium":    {"r": (1, 7),   "f": (5, 20),  "m": (500, 1200),
                           "aov": (500, 1000), "oc": (10, 35)},
            "Health":     {"r": (1, 14),  "f": (5, 15),  "m": (300, 700),
                           "aov": (300, 600),  "oc": (8, 25)},
            "Family":     {"r": (1, 10),  "f": (8, 20),  "m": (500, 1500),
                           "aov": (500, 1200), "oc": (12, 40)},
            "Occasional": {"r": (10, 45), "f": (1, 5),   "m": (200, 600),
                           "aov": (200, 500),  "oc": (1, 8)},
        }[seg]

        n_fav = np.random.randint(1, 4)
        fav = list(np.random.choice(CUISINES, size=n_fav, replace=False))

        rows.append({
            "user_id": f"U{i+1:05d}",
            "name": fake.name(),
            "city": city,
            "segment": seg,
            "dietary_preference": diet,
            "veg_days": json.dumps(veg_days),
            "rfm_recency": np.random.randint(*rfm_cfg["r"]),
            "rfm_frequency": np.random.randint(*rfm_cfg["f"]),
            "rfm_monetary": int(np.random.uniform(*rfm_cfg["m"])),
            "order_count": np.random.randint(*rfm_cfg["oc"]),
            "avg_order_value": int(np.random.uniform(*rfm_cfg["aov"])),
            "favourite_cuisines": json.dumps(fav),
        })
    return pd.DataFrame(rows)


def generate_order_history(users: pd.DataFrame,
                           restaurants: pd.DataFrame,
                           menu: pd.DataFrame,
                           n: int = NUM_ORDERS) -> pd.DataFrame:
    menu_by_rest: dict[str, pd.DataFrame] = {
        rid: grp for rid, grp in menu.groupby("restaurant_id")
    }
    rest_ids = restaurants["restaurant_id"].values
    rest_cuisines = dict(zip(restaurants["restaurant_id"],
                             restaurants["primary_cuisine"]))
    user_rows = users.to_dict("records")
    rows = []

    for i in range(n):
        u = random.choice(user_rows)
        fav = json.loads(u["favourite_cuisines"])

        # 70 % chance to pick a restaurant matching a favourite cuisine
        if np.random.random() < 0.70:
            matching = [rid for rid in rest_ids if rest_cuisines[rid] in fav]
            rid = random.choice(matching) if matching else random.choice(rest_ids)
        else:
            rid = random.choice(rest_ids)

        rmenu = menu_by_rest.get(rid)
        if rmenu is None or len(rmenu) == 0:
            continue

        seg = u["segment"]
        n_items = {"Budget": (1, 4), "Premium": (2, 6), "Health": (1, 4),
                   "Family": (3, 8), "Occasional": (1, 5)}[seg]
        k = min(np.random.randint(*n_items), len(rmenu))
        ordered = rmenu.sample(k)

        ts = random_timestamp()
        rows.append({
            "order_id": f"O{i+1:06d}",
            "user_id": u["user_id"],
            "restaurant_id": rid,
            "order_time": ts.isoformat(),
            "meal_period": meal_period(ts.hour),
            "items_ordered": json.dumps(ordered["item_id"].tolist()),
            "order_value": int(ordered["price"].sum()),
            "was_completed": bool(np.random.random() < 0.92),
        })
    return pd.DataFrame(rows)


def generate_sessions_and_events(
    users: pd.DataFrame,
    restaurants: pd.DataFrame,
    menu: pd.DataFrame,
    n: int = NUM_SESSIONS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    menu_by_rest: dict[str, pd.DataFrame] = {
        rid: grp for rid, grp in menu.groupby("restaurant_id")
    }
    rest_ids = restaurants["restaurant_id"].values
    rest_cuisines = dict(zip(restaurants["restaurant_id"],
                             restaurants["primary_cuisine"]))
    rest_zones = dict(zip(restaurants["restaurant_id"],
                          restaurants["zone"]))
    user_rows = users.to_dict("records")

    sess_rows: list[dict] = []
    evt_rows: list[dict] = []
    eid = 1

    for i in range(n):
        u = random.choice(user_rows)
        fav = json.loads(u["favourite_cuisines"])

        if np.random.random() < 0.60:
            matching = [rid for rid in rest_ids if rest_cuisines[rid] in fav]
            rid = random.choice(matching) if matching else random.choice(rest_ids)
        else:
            rid = random.choice(rest_ids)

        rmenu_full = menu_by_rest.get(rid)
        if rmenu_full is None or len(rmenu_full) == 0:
            continue

        ts = random_timestamp()
        mp = meal_period(ts.hour)
        zone = rest_zones.get(rid, np.random.choice(ZONES))

        # dietary toggle
        if np.random.random() < 0.12:
            if u["dietary_preference"] in ("veg", "vegan"):
                dtoggle = u["dietary_preference"]
            else:
                dtoggle = np.random.choice(["veg", "non-veg"], p=[0.40, 0.60])
        else:
            dtoggle = "none"

        veg_days = json.loads(u["veg_days"])
        if ts.weekday() in veg_days:
            dtoggle = "veg"

        # apply dietary filter
        rmenu = rmenu_full[rmenu_full["availability"] == True].copy()
        if dtoggle == "veg":
            rmenu = rmenu[rmenu["veg_flag"] == True]
        elif dtoggle == "vegan":
            rmenu = rmenu[rmenu["veg_flag"] == True]
        if len(rmenu) < 2:
            continue

        # ── simulate cart ────────────────────────────────
        cart_ids: set[str] = set()
        cart_cats: list[str] = []
        cart_prices: list[int] = []
        cpos = 0
        cur = ts

        # first organic add (prefer main / combo)
        mains = rmenu[rmenu["category"].isin(["main", "combo"])]
        first = mains.sample(1).iloc[0] if len(mains) > 0 else rmenu.sample(1).iloc[0]
        cart_ids.add(first["item_id"])
        cart_cats.append(first["category"])
        cart_prices.append(first["price"])
        cpos += 1
        cur += timedelta(seconds=int(np.random.randint(5, 30)))

        evt_rows.append({
            "event_id": f"E{eid:07d}",
            "session_id": f"S{i+1:06d}",
            "item_id": first["item_id"],
            "timestamp": cur.isoformat(),
            "cart_position": cpos,
            "was_recommendation": False,
            "was_accepted": None,
            "position_shown": None,
        })
        eid += 1

        # recommendation rounds
        n_rounds = np.random.randint(1, 4)
        for _ in range(n_rounds):
            cur += timedelta(seconds=int(np.random.randint(10, 60)))
            cands = rmenu[~rmenu["item_id"].isin(cart_ids)]
            if len(cands) < 3:
                break

            n_show = min(np.random.randint(8, 11), len(cands))
            shown = cands.sample(n_show)

            for pos, (_, rec) in enumerate(shown.iterrows(), 1):
                prob = 0.12
                prob *= 1.0 / (1.0 + 0.15 * (pos - 1))

                seg = u["segment"]
                if seg == "Budget" and rec["price"] > 200:
                    prob *= 0.50
                elif seg == "Premium" and rec["price"] < 80:
                    prob *= 0.70

                if rec["category"] in ("beverage", "dessert", "bread", "side"):
                    if rec["category"] not in cart_cats:
                        prob *= 1.50

                if rec["bestseller_flag"]:
                    prob *= 1.30

                accepted = bool(np.random.random() < prob)
                if accepted:
                    cpos += 1
                    cart_ids.add(rec["item_id"])
                    cart_cats.append(rec["category"])
                    cart_prices.append(rec["price"])

                evt_rows.append({
                    "event_id": f"E{eid:07d}",
                    "session_id": f"S{i+1:06d}",
                    "item_id": rec["item_id"],
                    "timestamp": cur.isoformat(),
                    "cart_position": cpos if accepted else None,
                    "was_recommendation": True,
                    "was_accepted": accepted,
                    "position_shown": pos,
                })
                eid += 1

            # occasional organic add between rounds
            if np.random.random() < 0.30:
                remaining = rmenu[~rmenu["item_id"].isin(cart_ids)]
                if len(remaining) > 0:
                    org = remaining.sample(1).iloc[0]
                    cpos += 1
                    cart_ids.add(org["item_id"])
                    cart_cats.append(org["category"])
                    cart_prices.append(org["price"])
                    cur += timedelta(seconds=int(np.random.randint(5, 20)))
                    evt_rows.append({
                        "event_id": f"E{eid:07d}",
                        "session_id": f"S{i+1:06d}",
                        "item_id": org["item_id"],
                        "timestamp": cur.isoformat(),
                        "cart_position": cpos,
                        "was_recommendation": False,
                        "was_accepted": None,
                        "position_shown": None,
                    })
                    eid += 1

        # completion
        comp_base = {"Budget": 0.72, "Premium": 0.85, "Health": 0.78,
                     "Family": 0.85, "Occasional": 0.65}[u["segment"]]
        if len(cart_ids) >= 3:
            comp_base += 0.10
        completed = bool(np.random.random() < min(comp_base, 0.95))
        final_val = int(sum(cart_prices)) if completed else 0

        sess_rows.append({
            "session_id": f"S{i+1:06d}",
            "user_id": u["user_id"],
            "restaurant_id": rid,
            "start_time": ts.isoformat(),
            "meal_period": mp,
            "zone": zone,
            "dietary_toggle": dtoggle,
            "order_completed": completed,
            "final_order_value": final_val,
            "num_cart_items": len(cart_ids),
        })

    return pd.DataFrame(sess_rows), pd.DataFrame(evt_rows)


# ────────────────────────────────────────────────────────────────
# VERIFICATION
# ────────────────────────────────────────────────────────────────

def verify(restaurants, menu, users, orders, sessions, events):
    sep = "=" * 62
    print(f"\n{sep}")
    print("  DATA VERIFICATION REPORT")
    print(sep)

    print(f"\nRestaurants : {len(restaurants)}")
    print(f"  Cuisines  : {restaurants['primary_cuisine'].value_counts().to_dict()}")
    print(f"  Tiers     : {restaurants['price_tier'].value_counts().to_dict()}")

    print(f"\nMenu Items  : {len(menu)}")
    print(f"  Categories: {menu['category'].value_counts().to_dict()}")
    veg_ct = int(menu['veg_flag'].sum())
    print(f"  Veg / Non-veg : {veg_ct} / {len(menu) - veg_ct}")
    print(f"  Combos    : {int(menu['is_combo'].sum())}")
    print(f"  Avg price : Rs {menu['price'].mean():.0f}")
    print(f"  Below 10% margin : {int((menu['margin_pct'] < 10).sum())}")
    print(f"  Out of stock     : {int((~menu['availability']).sum())}")

    print(f"\nUsers       : {len(users)}")
    print(f"  Segments  : {users['segment'].value_counts().to_dict()}")
    print(f"  Diet pref : {users['dietary_preference'].value_counts().to_dict()}")

    print(f"\nOrders      : {len(orders)}")
    comp_pct = orders["was_completed"].mean() * 100
    print(f"  Completed : {int(orders['was_completed'].sum())} ({comp_pct:.1f}%)")
    print(f"  Avg value : Rs {orders['order_value'].mean():.0f}")

    odt = pd.to_datetime(orders["order_time"])
    w3 = START_DATE + timedelta(weeks=3)
    w4 = START_DATE + timedelta(weeks=4)
    print(f"  Train (wk 1-3) : {int((odt < w3).sum())}")
    print(f"  Val   (wk 4)   : {int(((odt >= w3) & (odt < w4)).sum())}")
    print(f"  Test  (wk 5)   : {int((odt >= w4).sum())}")

    print(f"\nSessions    : {len(sessions)}")
    s_comp = sessions["order_completed"].mean() * 100
    print(f"  Completed : {int(sessions['order_completed'].sum())} ({s_comp:.1f}%)")
    print(f"  Avg items : {sessions['num_cart_items'].mean():.1f}")
    tog = int((sessions["dietary_toggle"] != "none").sum())
    print(f"  Diet toggle set : {tog}")

    print(f"\nCart Events  : {len(events)}")
    recs = events[events["was_recommendation"] == True]
    acc = recs[recs["was_accepted"] == True]
    rate = len(acc) / len(recs) * 100 if len(recs) > 0 else 0
    print(f"  Recommendations shown    : {len(recs)}")
    print(f"  Recommendations accepted : {len(acc)}")
    print(f"  Acceptance rate : {rate:.1f}%  (target 8-12%)")
    print(f"  Organic adds    : {int((events['was_recommendation'] == False).sum())}")

    print(f"\n{sep}")
    if 8 <= rate <= 12:
        print("  [OK] Acceptance rate within target range")
    else:
        print(f"  [!!] Acceptance rate {rate:.1f}% outside 8-12% target")
    print(sep + "\n")


# ────────────────────────────────────────────────────────────────
# MAIN
# ────────────────────────────────────────────────────────────────

def main():
    print("Step 1 - CSAO Synthetic Data Generation")
    print("-" * 42)

    print("[1/6] Generating restaurants …")
    restaurants = generate_restaurants()
    print(f"       {len(restaurants)} restaurants")

    print("[2/6] Generating menu items …")
    menu = generate_menu_items(restaurants)
    print(f"       {len(menu)} menu items")

    print("[3/6] Generating users …")
    users = generate_users()
    print(f"       {len(users)} users")

    print("[4/6] Generating order history …")
    orders = generate_order_history(users, restaurants, menu)
    print(f"       {len(orders)} orders")

    print("[5/6] Generating sessions & cart events …")
    sessions, events = generate_sessions_and_events(users, restaurants, menu)
    print(f"       {len(sessions)} sessions, {len(events)} cart events")

    print("[6/6] Saving to CSV …")
    restaurants.to_csv(os.path.join(DATA_DIR, "restaurants.csv"), index=False)
    menu.to_csv(os.path.join(DATA_DIR, "menu_items.csv"), index=False)
    users.to_csv(os.path.join(DATA_DIR, "users.csv"), index=False)
    orders.to_csv(os.path.join(DATA_DIR, "order_history.csv"), index=False)
    sessions.to_csv(os.path.join(DATA_DIR, "sessions.csv"), index=False)
    events.to_csv(os.path.join(DATA_DIR, "cart_events.csv"), index=False)
    print(f"       Saved to {DATA_DIR}/")

    verify(restaurants, menu, users, orders, sessions, events)


if __name__ == "__main__":
    main()
