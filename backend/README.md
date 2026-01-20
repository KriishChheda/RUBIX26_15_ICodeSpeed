# AI Proctor - Smart Examination Proctoring System Backend

This is the backend for the AI Proctoring system, designed to work seamlessly with a MongoDB database. The backend is built using Django and provides RESTful APIs for user authentication, test management, proctoring functionalities, and result tracking.

## Project Structure

The backend project is organized into several key components:

- **ai_proctor/**: Contains the main Django project settings and configuration files.
- **apps/**: Contains the various applications for authentication, test management, proctoring, and results.
  - **authentication/**: Handles user authentication and authorization.
  - **tests/**: Manages test creation, editing, and retrieval.
  - **proctoring/**: Implements proctoring features and AI integrations.
  - **results/**: Manages the storage and retrieval of test results.
- **middleware/**: Contains custom middleware for handling authentication and authorization.
- **utils/**: Contains utility functions, including MongoDB client interactions.
- **manage.py**: Command-line utility for managing the Django project.
- **requirements.txt**: Lists the required dependencies for the project.

## Features

- **User Authentication**: Role-based access control for admins and students.
- **Test Management**: Create, edit, and manage tests with various question types.
- **Proctoring**: Real-time monitoring and violation detection during tests.
- **Results Analytics**: Store and retrieve test results with detailed analytics.

## Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd backend
   ```

2. Create a virtual environment and activate it:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

4. Set up the MongoDB database and update the database settings in `ai_proctor/settings.py`.

5. Run database migrations:
   ```
   python manage.py migrate
   ```

6. Start the development server:
   ```
   python manage.py runserver
   ```

## Usage

- Access the API endpoints for authentication, test management, proctoring, and results through the defined URLs in the respective `urls.py` files within each app.

## Contributing

Feel free to contribute to this project by submitting issues or pull requests. This project is intended for educational and demonstration purposes.

## License

This project is licensed under the MIT License.