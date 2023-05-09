Fortune Teller App

This is a Dash app that allows users to visualize the historical stock prices and predict future prices of selected companies, as well as adjust the parameters of the prediction model and the confidence interval. The app is designed to be interactive, responsive, and easy to use.
The app is deployed via a hosting service called Heroku and the link to our standalone visualization can be found here- https://fortune-tellers.herokuapp.com/

Disclaimer-
Please note that due to API limitations, we can currently only query 5 tickers per minute. 
Also note that there is a gap between the last date recorded and the first day of forecast which represents the time difference between the closing time of that day (5 pm) and the opening bell of the next day (8 am).

Design Rationale

To design an intuitive and seamless user experience for exploring stock market data, we selected a line chart as the primary visualization for historical and predicted stock prices. This format is familiar and allows for easy comparison of trends and patterns across different companies. To enable users to select the time period of interest and the company they are interested in, we added a range slider and search menu. Additionally, the confidence interval is displayed as shaded areas around the predicted prices, providing users with a sense of the uncertainty of the predictions.

When deciding on the interactive elements for the app, we focused on creating an interface that was easy to use and navigate. We chose to use a search bar instead of a dropdown to allow users to search for a specific ticker. Dropdowns can become unwieldy and difficult to navigate when there are a large number of options available, whereas search bars allow for quick and easy selection of a specific ticker.

To enable users to change the range of the x-axis on the historical data plot, we opted for a slider instead of other options such as dropdowns or input fields. Sliders provide a visual representation of the data range, allowing users to easily adjust the range by dragging the slider handle.

In addition to the slider, we provided a calendar as an alternative way for users to select specific dates. This design decision was made to offer users more flexibility in selecting specific dates, rather than just selecting a range of years. The calendar also provides an intuitive and easy-to-use interface for selecting specific dates.

Lastly, we used radio buttons to allow users to select the confidence interval for the future price prediction plot. Radio buttons provide a clear and concise way for users to make a choice between a limited number of options. Furthermore, radio buttons can be easily formatted to provide visual feedback to the user, such as highlighting the selected option, improving the user experience. These design decisions work together to create a cohesive and intuitive user interface for exploring stock market data.






Development Process

Our team consisted of four members, who each contributed to different aspects of the app development:

Aditya: data preprocessing, app testing and debugging
Dorna: data visualization, user interface components, app testing and debugging
Meenal: data visualization, user interface components, documentation
Prithviraj: prediction model implementation, app testing and debugging

We spent approximately 45 people-hours in total developing the app, spread over a period of two weeks. The most time-consuming aspects of the development were:

During our development process, we encountered several challenges across various components of the project. In the data acquisition phase, we encountered a challenge in finding the most up-to-date and accurate data sources for our project. We had to explore multiple options and compare their quality and relevance to our project requirements. It took some time and effort to identify the most suitable sources that could provide the necessary data for our prediction model. However, through our thorough research and evaluation, we were eventually able to obtain the most recent and reliable datasets for our project.

Secondly, implementing the prediction model involved selecting the appropriate machine learning algorithms and optimizing performance and accuracy. This process required considerable effort and tuning to achieve the desired results.

As for the implementation of the prediction model, we decided to use the ARIMAl due to its ability to handle time-series data and capture the complex patterns that may exist in our data. We found that it was suitable for our project requirements and could produce accurate predictions for future trends.

Thirdly, designing the user interface was also challenging, as we relied solely on internal feedback and discussion to create an intuitive and user-friendly experience. Styling the app proved to be particularly difficult, and required extra attention to ensure a polished final product.

In terms of designing the user interface, we decided to use Dash Plotly, a Python library for creating web applications with interactive visualizations. We chose Dash Plotly because of its ease of use, flexibility, and ability to create dynamic and interactive charts that can be easily customized. It allowed us to create a visually appealing and user-friendly interface for our app, which we believed was crucial in ensuring a positive user experience.

Lastly, deployment and testing was the most difficult aspect of the project due to several challenges. We faced difficulties in hosting an interactive visualization made in Python on GitHub pages, and had to set up a server environment to overcome this. Additionally, we had to ensure compatibility with different browsers and devices, and resolve any issues that were reported by users.

Despite these challenges, the development process was a valuable learning experience that allowed us to hone our skills in data science, machine learning, web development, and teamwork. It also taught us the importance of incorporating user feedback and effectively collaborating as a team to produce a successful final product.
