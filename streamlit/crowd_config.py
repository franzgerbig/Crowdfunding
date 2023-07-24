"""

Config file for Streamlit App

"""

from members import Member


TITLE = "Give your crowdfunding a kick? Success prediction of crowdfunding campaigns on Kickstarter"

PROMOTION = "Bootcamp Data Scientist\nMay 2023"

TEAM_MEMBERS = [
    Member(
        name="Franz Gerbig",
        linkedin_url="https://www.linkedin.com/in/franzgerbig",
        github_url="https://github.com/franzgerbig"
    ),
    Member( 
        name="Hendrik Bosse",
        github_url="https://github.com/hebosse",
    )
]