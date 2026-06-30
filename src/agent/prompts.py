CLASSIFICATION_PROMPT = """
        Your are AI Laptop Sales Assistant.
        Analyze this customer message and classify it:

        User Message: {user_input}

        Provide:
        1. usage_profile: One of ["gaming", "student", "basic", "workstation"]
        2. user_emphasis: List of emphasized features from ["cpu_tier", "gpu_tier", "ram", "ssd_present", "price"], or null
        3. filters: Dictionary of specific requirements mentioned or null if none specified
        4. specific_game: If the user mentions a specific game title, extract it. 
        5. gibberish: set it to True, if user talking out of laptops context or random typing

        Only include filters that are explicitly mentioned by the user.
        """
MAPPING_PROMPT = """
                Given the following recommended system requirements for the game {game_name}:

                GPU: {gpu}
                RAM: {ram}
                
                Our dataset includes:
                GPUs: GTX 1650, RTX 2050, RTX 3050, RTX 3050 Ti, RTX 3060, RTX 3070, RTX 3070 Ti, RTX 3080, RTX 3080 Ti

                Extract and map these requirements to the following laptop specification format:
                - Only extract Nvidia (RTX/GTX) GPUs. If the requirement is worse than out minimum GPU (GTX 1650 in this case), set the value to GTX 1650.
                - For RTX and GTX GPUs, use the format: RTX 3060, GTX 1660, etc.
                - If the game needs specific hardware not in our dataset, choose the closest possible match for example (RTX 2070 -> RTX 3050)
                - If the game requires hardware worse than our minimum spec laptops, set the value to our minimum spec hardware (GTX 1650).

            """
EXPLANATION_PROMPT_GAME = """
            The following laptops have been recommended based on the user's need to play on reccomanded settings this game: {game_name}

            {reccomended_laptops}

            Please provide a detailed explanation of why these laptops are suitable for the user.

            Here is the context about the user:
            {user_context}

            Here is the context about the game  recommended system requirements:
            GPU: {gpu}\n
            CPU: {cpu}\n
            RAM: {ram}\n

            """
EXPLANATION_PROMPT = """
            The following laptops have been recommended based on the user's needs:

            {reccomended_laptops}

            Please provide a detailed explanation of why these laptops are suitable for the user.

            Here is the context about the user:
            {user_context}
            """
