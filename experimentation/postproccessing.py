import pandas as pd


def generate_latex_tables(csv_file):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_file)

    # Define the domains (to be used for grouping)
    domains = ['zenotravel', 'expedition', 'market_trader']
    domains_names = {
        'zenotravel': 'Zenotravel',
        'expedition': 'Settlers',
        'market_trader': 'Market Trader',

    }

    # Initialize a dictionary to store tables for each domain
    latex_tables = {}

    # Process each domain
    for domain in domains:
        # Filter the DataFrame for the current domain
        domain_data = df[df['name'].str.contains(domain)]
        has_sl = domain in ['expedition', 'zenotravel', 'grid']
        solved_with_sl = 0
        solved_without_sl = 0

        # Initialize an empty list to store the rows for the LaTeX table of this domain
        table_rows = []

        # Initialize a counter for problem numbers
        # Process the rows for the domain
        for i in range(0, len(domain_data), 1+int(has_sl)):  # We expect two rows per domain (with and without social law)
            row_without_social_law = domain_data.iloc[i + int(has_sl)]

            if has_sl:
                row_with_social_law = domain_data.iloc[i]
                row_without_social_law = domain_data.iloc[i + 1]

                # Extract the necessary columns
                time_solved_with_sl = round(row_with_social_law['time'], 2) if row_with_social_law[
                                                                                 'result'] != 'SocialLawRobustnessStatus.UNKNOWN' else '-'
            else:
                time_solved_with_sl = '-'
            time_solved_without_sl = round(row_without_social_law['time'], 2) if row_without_social_law[
                                                                           'result'] != 'SocialLawRobustnessStatus.UNKNOWN' else '-'
            prob_name = row_without_social_law['name'].split('_')[-1]
            # Create the LaTeX row for this problem
            table_row = f"{prob_name} & {time_solved_with_sl} & {time_solved_without_sl} \\\\"
            table_rows.append(table_row)
            solved_with_sl += int(time_solved_with_sl != '-')
            solved_without_sl += int(time_solved_without_sl != '-')
        coverage_line = f"\\hline\nCoverage & {solved_with_sl} & {solved_without_sl} \\\\"
        table_rows.append(coverage_line)
        # Create the final LaTeX table for this domain
        header = "\\begin{tabular}{| c | c | c |}\n\\hline\n"+domains_names[domain]+" & With Sl & Without SL \\\\\n\\hline "

        footer = "\n\\hline\n\\end{tabular}"

        # Join header, rows, and footer together
        latex_table = header + "\n" + "\n".join(table_rows) + footer

        # Store the table for the domain
        latex_tables[domain] = latex_table

        # Store the table for the domain
        latex_tables[domain] = latex_table

    return latex_tables


# Example usage:
csv_file = '/experimentation/important_logs/experiment_log_Jan-21-2025.csv'  # Replace with the path to your CSV file
latex_tables = generate_latex_tables(csv_file)

# Print LaTeX tables for each domain
for domain, latex_table in latex_tables.items():
    #print(f"Table for {domain} domain:\n")
    print(latex_table)
    print("\n")
