package org.example;

import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.impl.model.file.FileDataModel;
import org.apache.mahout.cf.taste.impl.neighborhood.ThresholdUserNeighborhood;
import org.apache.mahout.cf.taste.impl.recommender.GenericUserBasedRecommender;
import org.apache.mahout.cf.taste.impl.recommender.GenericItemBasedRecommender;
import org.apache.mahout.cf.taste.impl.similarity.PearsonCorrelationSimilarity;
import org.apache.mahout.cf.taste.impl.similarity.LogLikelihoodSimilarity;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.neighborhood.UserNeighborhood;
import org.apache.mahout.cf.taste.recommender.RecommendedItem;
import org.apache.mahout.cf.taste.recommender.Recommender;
import org.apache.mahout.cf.taste.similarity.ItemSimilarity;
import org.apache.mahout.cf.taste.similarity.UserSimilarity;

import java.io.*;
import java.util.*;

/**
 * AI-Based Recommendation System using Apache Mahout
 * Supports both collaborative filtering and content-based recommendations
 */
public class RecommendationSystem {

    private DataModel dataModel;
    private Recommender userBasedRecommender;
    private Recommender itemBasedRecommender;
    private Map<Long, Product> productCatalog;
    private Map<Long, User> userProfiles;

    // Product class to store product information
    static class Product {
        long id;
        String name;
        String category;
        double price;
        Set<String> tags;

        public Product(long id, String name, String category, double price, Set<String> tags) {
            this.id = id;
            this.name = name;
            this.category = category;
            this.price = price;
            this.tags = tags;
        }

        @Override
        public String toString() {
            return String.format("Product{id=%d, name='%s', category='%s', price=%.2f}",
                    id, name, category, price);
        }
    }

    // User class to store user preferences
    static class User {
        long id;
        String name;
        Set<String> preferredCategories;
        double priceRangeMin;
        double priceRangeMax;

        public User(long id, String name, Set<String> preferredCategories,
                    double priceRangeMin, double priceRangeMax) {
            this.id = id;
            this.name = name;
            this.preferredCategories = preferredCategories;
            this.priceRangeMin = priceRangeMin;
            this.priceRangeMax = priceRangeMax;
        }
    }

    public RecommendationSystem() {
        initializeSampleData();
    }

    /**
     * Initialize sample data for demonstration
     */
    private void initializeSampleData() {
        // Initialize product catalog
        productCatalog = new HashMap<>();
        productCatalog.put(1L, new Product(1, "MacBook Pro", "Electronics", 1299.99,
                new HashSet<>(Arrays.asList("laptop", "apple", "professional"))));
        productCatalog.put(2L, new Product(2, "iPhone 15", "Electronics", 999.99,
                new HashSet<>(Arrays.asList("phone", "apple", "mobile"))));
        productCatalog.put(3L, new Product(3, "Nike Air Max", "Footwear", 129.99,
                new HashSet<>(Arrays.asList("shoes", "nike", "sports"))));
        productCatalog.put(4L, new Product(4, "The Great Gatsby", "Books", 12.99,
                new HashSet<>(Arrays.asList("fiction", "classic", "literature"))));
        productCatalog.put(5L, new Product(5, "Wireless Headphones", "Electronics", 199.99,
                new HashSet<>(Arrays.asList("audio", "wireless", "headphones"))));
        productCatalog.put(6L, new Product(6, "Running Shorts", "Clothing", 29.99,
                new HashSet<>(Arrays.asList("sports", "clothing", "running"))));
        productCatalog.put(7L, new Product(7, "Coffee Maker", "Appliances", 79.99,
                new HashSet<>(Arrays.asList("coffee", "appliance", "kitchen"))));
        productCatalog.put(8L, new Product(8, "Yoga Mat", "Sports", 24.99,
                new HashSet<>(Arrays.asList("yoga", "fitness", "exercise"))));

        // Initialize user profiles
        userProfiles = new HashMap<>();
        userProfiles.put(1L, new User(1, "Alice",
                new HashSet<>(Arrays.asList("Electronics", "Books")), 10.0, 500.0));
        userProfiles.put(2L, new User(2, "Bob",
                new HashSet<>(Arrays.asList("Sports", "Clothing")), 20.0, 200.0));
        userProfiles.put(3L, new User(3, "Charlie",
                new HashSet<>(Arrays.asList("Electronics", "Appliances")), 50.0, 1500.0));
        userProfiles.put(4L, new User(4, "Diana",
                new HashSet<>(Arrays.asList("Books", "Sports")), 15.0, 300.0));

        // Create sample ratings data
        createSampleRatingsFile();
    }

    /**
     * Create sample ratings file for Mahout
     */
    private void createSampleRatingsFile() {
        try (PrintWriter writer = new PrintWriter(new FileWriter("ratings.csv"))) {
            // Format: userID,itemID,rating
            writer.println("1,1,5.0");  // Alice rates MacBook Pro highly
            writer.println("1,2,4.0");  // Alice rates iPhone well
            writer.println("1,4,5.0");  // Alice loves The Great Gatsby
            writer.println("1,5,4.0");  // Alice likes wireless headphones

            writer.println("2,3,5.0");  // Bob loves Nike Air Max
            writer.println("2,6,4.0");  // Bob likes running shorts
            writer.println("2,8,5.0");  // Bob loves yoga mat
            writer.println("2,1,2.0");  // Bob doesn't like expensive electronics

            writer.println("3,1,5.0");  // Charlie loves MacBook Pro
            writer.println("3,2,4.0");  // Charlie likes iPhone
            writer.println("3,5,5.0");  // Charlie loves wireless headphones
            writer.println("3,7,4.0");  // Charlie likes coffee maker

            writer.println("4,4,5.0");  // Diana loves The Great Gatsby
            writer.println("4,8,4.0");  // Diana likes yoga mat
            writer.println("4,3,3.0");  // Diana moderately likes Nike shoes
            writer.println("4,6,3.0");  // Diana moderately likes running shorts

        } catch (IOException e) {
            System.err.println("Error creating ratings file: " + e.getMessage());
        }
    }

    /**
     * Initialize the recommendation engines
     */
    public void initializeRecommenders() throws TasteException, IOException {
        // Load data model from file
        dataModel = new FileDataModel(new File("ratings.csv"));

        // Initialize User-Based Collaborative Filtering
        UserSimilarity userSimilarity = new PearsonCorrelationSimilarity(dataModel);
        UserNeighborhood neighborhood = new ThresholdUserNeighborhood(0.1, userSimilarity, dataModel);
        userBasedRecommender = new GenericUserBasedRecommender(dataModel, neighborhood, userSimilarity);

        // Initialize Item-Based Collaborative Filtering
        ItemSimilarity itemSimilarity = new LogLikelihoodSimilarity(dataModel);
        itemBasedRecommender = new GenericItemBasedRecommender(dataModel, itemSimilarity);

        System.out.println("Recommendation engines initialized successfully!");
    }

    /**
     * Get user-based collaborative filtering recommendations
     */
    public List<RecommendedItem> getUserBasedRecommendations(long userId, int numRecommendations)
            throws TasteException {
        return userBasedRecommender.recommend(userId, numRecommendations);
    }

    /**
     * Get item-based collaborative filtering recommendations
     */
    public List<RecommendedItem> getItemBasedRecommendations(long userId, int numRecommendations)
            throws TasteException {
        return itemBasedRecommender.recommend(userId, numRecommendations);
    }

    /**
     * Content-based filtering using user preferences and product attributes
     */
    public List<Product> getContentBasedRecommendations(long userId, int numRecommendations) {
        User user = userProfiles.get(userId);
        if (user == null) {
            return new ArrayList<>();
        }

        List<Product> recommendations = new ArrayList<>();

        // Get products that match user preferences
        for (Product product : productCatalog.values()) {
            if (isProductSuitableForUser(product, user)) {
                recommendations.add(product);
            }
        }

        // Sort by relevance (simple scoring based on category match and price preference)
        recommendations.sort((p1, p2) -> {
            double score1 = calculateContentScore(p1, user);
            double score2 = calculateContentScore(p2, user);
            return Double.compare(score2, score1); // Descending order
        });

        return recommendations.subList(0, Math.min(numRecommendations, recommendations.size()));
    }

    /**
     * Check if product is suitable for user based on preferences
     */
    private boolean isProductSuitableForUser(Product product, User user) {
        // Check if product is in user's preferred categories
        boolean categoryMatch = user.preferredCategories.contains(product.category);

        // Check if product is within user's price range
        boolean priceMatch = product.price >= user.priceRangeMin && product.price <= user.priceRangeMax;

        return categoryMatch && priceMatch;
    }

    /**
     * Calculate content-based score for product-user pair
     */
    private double calculateContentScore(Product product, User user) {
        double score = 0.0;

        // Category preference bonus
        if (user.preferredCategories.contains(product.category)) {
            score += 10.0;
        }

        // Price preference bonus (closer to middle of range gets higher score)
        double priceRangeMiddle = (user.priceRangeMin + user.priceRangeMax) / 2;
        double priceDistance = Math.abs(product.price - priceRangeMiddle);
        double maxDistance = (user.priceRangeMax - user.priceRangeMin) / 2;
        score += (1.0 - (priceDistance / maxDistance)) * 5.0;

        return score;
    }

    /**
     * Hybrid recommendation combining collaborative and content-based filtering
     */
    public List<String> getHybridRecommendations(long userId, int numRecommendations) throws TasteException {
        List<String> hybridRecommendations = new ArrayList<>();

        // Get collaborative filtering recommendations
        List<RecommendedItem> userBased = getUserBasedRecommendations(userId, numRecommendations / 2);
        List<RecommendedItem> itemBased = getItemBasedRecommendations(userId, numRecommendations / 2);

        // Get content-based recommendations
        List<Product> contentBased = getContentBasedRecommendations(userId, numRecommendations / 2);

        // Combine recommendations
        Set<Long> recommendedIds = new HashSet<>();

        // Add user-based recommendations
        for (RecommendedItem item : userBased) {
            long itemId = item.getItemID();
            if (!recommendedIds.contains(itemId)) {
                Product product = productCatalog.get(itemId);
                if (product != null) {
                    hybridRecommendations.add(String.format("User-Based: %s (Score: %.2f)",
                            product.name, item.getValue()));
                    recommendedIds.add(itemId);
                }
            }
        }

        // Add item-based recommendations
        for (RecommendedItem item : itemBased) {
            long itemId = item.getItemID();
            if (!recommendedIds.contains(itemId)) {
                Product product = productCatalog.get(itemId);
                if (product != null) {
                    hybridRecommendations.add(String.format("Item-Based: %s (Score: %.2f)",
                            product.name, item.getValue()));
                    recommendedIds.add(itemId);
                }
            }
        }

        // Add content-based recommendations
        for (Product product : contentBased) {
            if (!recommendedIds.contains(product.id)) {
                hybridRecommendations.add(String.format("Content-Based: %s", product.name));
                recommendedIds.add(product.id);
            }
        }

        return hybridRecommendations.subList(0, Math.min(numRecommendations, hybridRecommendations.size()));
    }

    /**
     * Display user information and preferences
     */
    public void displayUserInfo(long userId) {
        User user = userProfiles.get(userId);
        if (user != null) {
            System.out.println("\n--- User Information ---");
            System.out.println("User ID: " + user.id);
            System.out.println("Name: " + user.name);
            System.out.println("Preferred Categories: " + user.preferredCategories);
            System.out.println("Price Range: $" + user.priceRangeMin + " - $" + user.priceRangeMax);
        }
    }

    /**
     * Display product catalog
     */
    public void displayProductCatalog() {
        System.out.println("\n--- Product Catalog ---");
        for (Product product : productCatalog.values()) {
            System.out.println(product);
        }
    }

    /**
     * Main method to demonstrate the recommendation system
     */
    public static void main(String[] args) {
        RecommendationSystem system = new RecommendationSystem();

        try {
            // Initialize recommenders
            system.initializeRecommenders();

            // Display product catalog
            system.displayProductCatalog();

            // Test recommendations for each user
            for (long userId = 1; userId <= 4; userId++) {
                system.displayUserInfo(userId);

                System.out.println("\n--- Recommendations for User " + userId + " ---");

                // User-based collaborative filtering
                System.out.println("\nUser-Based Collaborative Filtering:");
                List<RecommendedItem> userBased = system.getUserBasedRecommendations(userId, 3);
                for (RecommendedItem item : userBased) {
                    Product product = system.productCatalog.get(item.getItemID());
                    if (product != null) {
                        System.out.printf("  %s (Score: %.2f)\n", product.name, item.getValue());
                    }
                }

                // Item-based collaborative filtering
                System.out.println("\nItem-Based Collaborative Filtering:");
                List<RecommendedItem> itemBased = system.getItemBasedRecommendations(userId, 3);
                for (RecommendedItem item : itemBased) {
                    Product product = system.productCatalog.get(item.getItemID());
                    if (product != null) {
                        System.out.printf("  %s (Score: %.2f)\n", product.name, item.getValue());
                    }
                }

                // Content-based filtering
                System.out.println("\nContent-Based Filtering:");
                List<Product> contentBased = system.getContentBasedRecommendations(userId, 3);
                for (Product product : contentBased) {
                    System.out.println("  " + product.name);
                }

                // Hybrid recommendations
                System.out.println("\nHybrid Recommendations:");
                List<String> hybrid = system.getHybridRecommendations(userId, 5);
                for (String recommendation : hybrid) {
                    System.out.println("  " + recommendation);
                }

                System.out.println("\n" + "=".repeat(50));
            }

        } catch (Exception e) {
            System.err.println("Error in recommendation system: " + e.getMessage());
            e.printStackTrace();
        }
    }
}
