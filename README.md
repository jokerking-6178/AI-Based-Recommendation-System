# AI-Based-Recommendation-System

#Company - Codtech IT Solutions

#Name - Shubham Garg

#Intern-ID - CT04DH1346

#Domain - Java Developer

#Duration - 4 weeks

#Mentor - Neela Santosh

# Project Description

AI-Based Recommendation System - Project Description
This project implements a comprehensive recommendation engine using Java and Apache Mahout, demonstrating multiple machine learning approaches to suggest products based on user preferences and behavior patterns.
System Overview
The recommendation system employs four distinct algorithms to provide personalized product suggestions. User-based collaborative filtering identifies similar users through Pearson correlation analysis and recommends items highly rated by comparable users. Item-based collaborative filtering uses log-likelihood similarity to find products frequently purchased together, suggesting items based on co-occurrence patterns. Content-based filtering matches products to user preferences by analyzing attributes like category, price range, and product tags. Finally, a hybrid approach combines all three methods to maximize recommendation accuracy and handle edge cases like new users or sparse data.
Technical Implementation
Built with Apache Mahout's machine learning framework, the system processes user-item ratings from CSV files to train collaborative filtering models. The architecture uses object-oriented design with dedicated classes for users and products, storing data in HashMaps for efficient O(1) lookups. Sample data includes eight diverse products (electronics, books, sports equipment) across different price ranges ($12.99 to $1299.99) and four user profiles with varying preferences and budgets.
Key Features
The system generates recommendations through multiple pathways: collaborative filtering identifies patterns in user behavior, content-based filtering ensures recommendations match stated preferences, and the hybrid model combines approaches for robust performance. Real-time scoring algorithms calculate product relevance based on category matches and price preferences, while similarity metrics determine user and item relationships.
Data and Validation
Sample data simulates realistic e-commerce scenarios with users like tech enthusiasts, sports lovers, and budget-conscious readers. The ratings dataset (1.0-5.0 scale) enables the system to learn user preferences and demonstrate how different algorithms produce varied but complementary recommendations.


# OUTPUT

<img width="1709" height="867" alt="Image" src="https://github.com/user-attachments/assets/1869fd5c-ad56-4951-bce0-be985e77168e" />

<img width="1419" height="808" alt="Image" src="https://github.com/user-attachments/assets/a5a63d7d-2faf-47bd-8c46-5f183e0f58d3" />

<img width="1166" height="798" alt="Image" src="https://github.com/user-attachments/assets/be0d6e42-523b-4f83-8f2e-0dcdb467a2aa" />

<img width="1336" height="757" alt="Image" src="https://github.com/user-attachments/assets/dca497e9-64bb-4899-879e-0a58f9e244ce" />
