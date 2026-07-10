interface QueryFilters {
  gender: string | null;
  category: string | null;
  brand: string | null;
  price: {
    gte?: number;
    lte?: number;
  };
  rating: {
    gte?: number;
    lte?: number;
  };
  keywords: string[];
}

export function extractFilters(query: string): QueryFilters {
  query = query.toLowerCase();
  const filters: QueryFilters = {
    gender: null,
    category: null,
    brand: null,
    price: {},
    rating: {},
    keywords: []
  };

  // Gender
  if (query.includes("men")) filters.gender = "men";
  if (query.includes("women")) filters.gender = "women";
  if (query.includes("kids")) filters.gender = "kids";

  // Brand detection
  const brands = [
    "samsung", "apple", "iphone", "xiaomi", "redmi", "oneplus", "oppo", "vivo", "realme",
    "nike", "adidas", "puma", "reebok", "under armour", "new balance",
    "titan", "fastrack", "casio", "fossil", "rolex", "omega",
    "ray-ban", "oakley", "prada", "gucci", "versace", "dolce gabbana",
    "hp", "dell", "lenovo", "asus", "acer", "msi", "apple macbook",
    "lg", "sony", "panasonic", "whirlpool", "bosch", "siemens",
    "puma", "adidas", "nike", "reebok", "under armour", "new balance",
    "boat", "jbl", "sony", "bose", "sennheiser", "audio technica",
    "canon", "nikon", "sony", "fujifilm", "gopro", "dji"
  ];
  
  for (const brand of brands) {
    if (query.includes(brand)) {
      filters.brand = brand;
      break;
    }
  }

  // Category keywords (expandable list)
  const categories = [
    // Main categories from dataset
    "clothing", "jewellery", "footwear", "mobile phones", "mobiles & accessories", 
    "automotive", "home decor", "home decor & festive needs", "home furnishing", "computers",
    "furniture", "electronics", "appliances", "sports", "books", "toys", "beauty",
    // Sub-categories and specific products
    "shoes", "tops", "mobiles", "jeans", "sarees", "laptops",
    "sneakers", "t-shirt", "laptop bag", "watch", "headphone", "speaker",
    "camera", "refrigerator", "washing machine", "air conditioner", "television",
    "books", "perfume", "wallet", "sunglasses", "backpack", "power bank",
    "keyboard", "mouse", "monitor", "printer", "router", "smartwatch",
    "earbuds", "trimmer", "hair dryer", "mixer grinder", "pressure cooker",
    "cookware", "bedsheet", "curtains", "sofa", "dining table", "wardrobe",
    "mattress", "pillow", "blanket", "water bottle", "coffee maker", "air fryer",
    "gas stove", "dinner set", "food processor", "juicer", "vacuum cleaner",
    "iron box", "stabilizer", "running shoes", "dumbbell", "yoga mat",
    "cricket bat", "football", "badminton racquet", "cycling", "gym bag",
    "protein powder", "fitness tracker", "gaming console", "gaming headset",
    "gaming mouse", "mechanical keyboard", "4k monitor", "led strip lights",
    "skincare", "moisturizer", "face wash", "sunscreen", "shampoo", "conditioner",
    "body lotion", "deodorant", "makeup kit", "foundation", "lipstick", "kajal",
    "nail polish", "hair oil", "hair colour", "electric toothbrush", "grooming kit",
    "epilator", "bookshelf", "study table", "bluetooth headphones", "iphone charger",
    "tripod", "kids toys", "kitchen appliances", "gardening tools",
    "musical instruments", "sports equipment", "travel accessories", "luggage",
    "art supplies", "craft supplies", "hobby kits", "drones",
    "car accessories", "bike accessories", "pet supplies", "health supplements",
    "face masks", "hand sanitizers", "backpacks", "bags", "laptop bags",
    // Furniture specific categories
    "chair", "table", "bed", "sofa set", "dining set", "office chair",
    "study chair", "computer chair", "gaming chair", "recliner", "bean bag",
    "stool", "bench", "ottoman", "footstool", "side table", "coffee table",
    "dining table", "study table", "office desk", "computer desk", "gaming desk",
    "bed frame", "bunk bed", "trundle bed", "day bed", "futon", "mattress",
    "wardrobe", "almirah", "cupboard", "bookshelf", "shelf", "rack",
    "chest of drawers", "dresser", "mirror", "wall mirror", "floor mirror",
    "shoe rack", "shoe cabinet", "tv unit", "tv stand", "entertainment unit",
    "showcase", "display cabinet", "filing cabinet", "locker", "storage unit"
  ];
  
  for (const cat of categories) {
    if (query.includes(cat)) {
      filters.category = cat;
      break;
    }
  }

  // Price
  const priceUnderMatch = query.match(/(?:under|less than|below)\s*(\d+)/);
  const priceAboveMatch = query.match(/(?:above|greater than|more than)\s*(\d+)/);
  const priceRangeMatch = query.match(/(\d+)\s*(?:to|-)\s*(\d+)/);
  const priceExactMatch = query.match(/at\s*(\d+)/);

  if (priceUnderMatch) filters.price.lte = parseInt(priceUnderMatch[1]);
  if (priceAboveMatch) filters.price.gte = parseInt(priceAboveMatch[1]);
  if (priceRangeMatch) {
    filters.price.gte = parseInt(priceRangeMatch[1]);
    filters.price.lte = parseInt(priceRangeMatch[2]);
  }
  if (priceExactMatch) {
    filters.price.gte = parseInt(priceExactMatch[1]);
    filters.price.lte = parseInt(priceExactMatch[1]);
  }

  // Rating
  const ratingAboveMatch = query.match(/(?:above|greater than|more than)\s*(\d(?:\.\d)?)\s*stars?/);
  const ratingExactMatch = query.match(/(\d(?:\.\d)?)\s*stars?/);
  const ratingUpToMatch = query.match(/(?:up to|below)\s*(\d(?:\.\d)?)\s*stars?/);

  if (ratingAboveMatch) filters.rating.gte = parseFloat(ratingAboveMatch[1]);
  else if (ratingExactMatch) filters.rating.gte = parseFloat(ratingExactMatch[1]);
  if (ratingUpToMatch) filters.rating.lte = parseFloat(ratingUpToMatch[1]);

  // Keyword tokens (remove known words and numbers)
  const removeWords = [
    "men", "women", "kids", "under", "above", "below", "less", "more", "than",
    "with", "stars", "to", "at", "between", "and", "up"
  ];

  // Start with all words, then filter out stop words and numbers
  let keywords = query.split(" ").filter(w => w.trim() !== '');

  // We don't remove the category from keywords anymore.
  // The search query needs it for relevance.
  filters.keywords = keywords
    .filter(w => !removeWords.includes(w) && isNaN(Number(w)));

  return filters;
}

export function isTopRatedQuery(query: string): boolean {
  const topRatedKeywords = ['top rated', 'best rated', 'highest rated', 'top', 'best'];
  return topRatedKeywords.some(keyword => query.toLowerCase().includes(keyword));
}

export function extractRatingFromQuery(query: string): { lt?: number; gte?: number } | null {
  const lowerQuery = query.toLowerCase();
  const underMatch = lowerQuery.match(/rating\s+(under|below|less\s+than)\s+(\d+(?:\.\d+)?)/);
  if (underMatch) return { lt: parseFloat(underMatch[2]) };
  
  const aboveMatch = lowerQuery.match(/rating\s+(above|over|greater\s+than)\s+(\d+(?:\.\d+)?)/);
  if (aboveMatch) return { gte: parseFloat(aboveMatch[2]) };

  const starAboveMatch = lowerQuery.match(/(\d+(?:\.\d+)?)\s+(star|rating)\s+(and\s+)?above/);
  if (starAboveMatch) return { gte: parseFloat(starAboveMatch[1]) };

  const exactStarMatch = lowerQuery.match(/(\d+(?:\.\d+)?)\s+star(?!s)/);
  if (exactStarMatch) return { gte: parseFloat(exactStarMatch[1]) };
  
  return null;
} 