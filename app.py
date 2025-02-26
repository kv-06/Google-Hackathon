from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

model = joblib.load("model/random_forest_customer_model.joblib")

FEATURE_COLUMNS = [
    "days_since_last_visit", "total_purchases", "no_of_items_in_cart", "frequency_of_visits",
    "purchase_frequency", "cart_value", "time_spent_on_site (mins)", "discount_activity"
]

CUSTOMER_TYPE_MAP = {
    0: "Cart Abandoner",
    1: "Frequent Buyer",
    2: "Inactive Customer",
    3: "New User"
}

EMAIL_TEMPLATES = {
    "Cart Abandoner": """
        <h3>Come Back and Grab Your Cart!</h3>
        <p>We noticed you left some items in your cart. Complete your purchase today and enjoy an exclusive discount!</p>
        <p><strong>Special Offer:</strong> 10% off if you complete your order now.</p>
        <a href='#' style='background: #ff6600; color: white; padding: 10px; text-decoration: none;'>Complete Your Purchase</a>
    """,
    "Frequent Buyer": """
        <h3>Thank You for Being a Valued Customer!</h3>
        <p>We appreciate your loyalty. Here’s an exclusive early access to our upcoming sale.</p>
        <p><strong>Special Offer:</strong> Get an extra 15% off on your next order.</p>
        <a href='#' style='background: #007bff; color: white; padding: 10px; text-decoration: none;'>Shop Now</a>
    """,
    "Inactive Customer": """
        <h3>We Miss You!</h3>
        <p>It’s been a while since you visited. We have some exciting new products waiting for you.</p>
        <p><strong>Come back today and enjoy a 20% discount on your next order!</strong></p>
        <a href='#' style='background: #28a745; color: white; padding: 10px; text-decoration: none;'>Browse New Arrivals</a>
    """,
    "New User": """
        <h3>Welcome to Our Store!</h3>
        <p>We’re so glad to have you here. Start your shopping journey with a special discount.</p>
        <p><strong>Exclusive Offer:</strong> 10% off on your first purchase!</p>
        <a href='#' style='background: #dc3545; color: white; padding: 10px; text-decoration: none;'>Start Shopping</a>
    """
}

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    email_content = None

    if request.method == "POST":
        try:
            user_input = {col: float(request.form[col].strip()) for col in FEATURE_COLUMNS}
            product = request.form.get("product", "").strip()
            if not product:
                return "Error: Product field is required"

            input_df = pd.DataFrame([user_input], columns=FEATURE_COLUMNS)
            predicted_class = model.predict(input_df)[0]
            prediction = CUSTOMER_TYPE_MAP.get(predicted_class, "Frequent Buyer")

            email_content = EMAIL_TEMPLATES.get(prediction, "")

        except ValueError:
            return "Error: Invalid input. Ensure all fields are filled correctly."
        except Exception as e:
            return f"Error: {str(e)}"

    return render_template("index.html", prediction=prediction, email_content=email_content)

if __name__ == "__main__":
    app.run(debug=True)
