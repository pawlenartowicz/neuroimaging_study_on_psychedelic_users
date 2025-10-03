# Load required packages
library(dplyr)
library(tidyr)
library(lmerTest)
library(emmeans)
library(ggplot2)
library(ggsignif)
library(effectsize)
library(bestNormalize)
library(reshape2)


# Define folder paths per city (update base path as needed)
folder_paths <- list(
  Wwa = "/path/to/project/LZs/Wwa",
  Krk = "/path/to/project/LZs/Krk"
)

# Function to process individual LZ text files per city
process_files <- function(folder_path, city_name) {
  files <- list.files(folder_path, pattern = "\\.txt$", full.names = TRUE)
  data_list <- list()
  
  for (file in files) {
    file_name <- basename(file)
    parts <- strsplit(file_name, "_")[[1]]
    subject_id <- parts[3]
    condition <- ifelse(parts[5] == "10.txt", "EO", "EC")
    
    data <- read.table(file, header = FALSE)
    
    mean_data <- data %>%
      rowMeans(na.rm = TRUE) %>%
      data.frame(Mean_LZ = .) %>%
      mutate(
        Trial_N = row_number(),
        Subject_ID = as.factor(subject_id),
        Condition = condition,
        City = city_name
      )
    
    data_list[[length(data_list) + 1]] <- mean_data
  }
  
  combined_data <- bind_rows(data_list)
  return(combined_data)
}

# Combine LZ data from both cities
LZ_df <- bind_rows(
  process_files(folder_paths$Wwa, "Wwa"),
  process_files(folder_paths$Krk, "Krk")
)

# Import participant metadata (update path as needed)
participants_data <- read.table("/path/to/project/Participants_data.csv", sep = "\t", header = TRUE)

# Align subject IDs for join
LZ_df <- LZ_df %>%
  mutate(Subject_ID = as.numeric(as.character(Subject_ID)))

participants_data <- participants_data %>%
  mutate(Subj.ID = as.numeric(Subj.ID))

# Merge LZ data with participant metadata
merged_data <- LZ_df %>%
  left_join(participants_data, by = c("Subject_ID" = "Subj.ID"))

# Linear mixed-effects model: Mean_LZ by Group, Condition, City, random intercept for Subject_ID
LZ_lmer <- lmer(Mean_LZ ~ Group * Condition + City.y + (1 | Subject_ID), data = merged_data)

# Display ANOVA table and effect sizes
anova_results <- anova(LZ_lmer)
print(anova_results)
print(eta_squared(LZ_lmer))

# Post-hoc contrasts with Holm adjustment
cons1_LZ <- emmeans(LZ_lmer, pairwise ~ Condition, adjust = "holm")$contrasts
cons2_LZ <- emmeans(LZ_lmer, pairwise ~ City.y, adjust = "holm")$contrasts
cons3_LZ <- emmeans(LZ_lmer, pairwise ~ Group | Condition, adjust = "holm", pbkrtest.limit = 16024)$contrasts

print(cons1_LZ)
print(cons2_LZ)
print(cons3_LZ)

# Prepare data for plotting
merged_data$Group[merged_data$Group == "Experimental"] <- "Users"
merged_data$Group[merged_data$Group == "Control"] <- "Non-users"
merged_data$Group <- factor(merged_data$Group, levels = c("Non-users", "Users"))

# Create prediction grid for plot
new_data_LZ <- expand.grid(
  Condition = c("EO", "EC"),
  Group = c("Users", "Non-users"),
  Mean_LZ = 0
)

# Fit model with random intercept and interaction term for prediction
LZ_lmer2 <- lmer(Mean_LZ ~ Condition * Group + (1 | Subject_ID), merged_data)

new_data_LZ$Mean_LZ <- predict(LZ_lmer2, new_data_LZ, re.form = NA)

# Calculate confidence intervals for predicted means
mm <- model.matrix(terms(LZ_lmer2), new_data_LZ)
pvar1 <- diag(mm %*% tcrossprod(vcov(LZ_lmer2), mm))

new_data_LZ <- data.frame(
  new_data_LZ,
  plo = new_data_LZ$Mean_LZ - sqrt(pvar1),
  phi = new_data_LZ$Mean_LZ + sqrt(pvar1)
)

# Plot bar graph with error bars and significance annotations
p_LZ <- ggplot(new_data_LZ, aes(x = Condition, y = Mean_LZ, fill = Group)) +
  geom_bar(stat = "identity", lwd = 1.2, color = "black", position = position_dodge()) +
  labs(x = "Condition", y = "LZ Complexity") +
  geom_errorbar(aes(ymin = plo, ymax = phi), width = 0.2, lwd = 1.2, position = position_dodge(0.9)) +
  theme_classic(base_size = 25) +
  theme(
    axis.title.y = element_text(margin = margin(t = 0, r = 20, b = 0, l = 0)),
    axis.title.x = element_text(margin = margin(t = 20, r = 0, b = 0, l = 0)),
    legend.key.size = unit(1, "cm"),
    plot.title = element_text(hjust = 0.5)
  ) +
  scale_fill_manual(values = c("#6047ff", "#fc6d6d"), name = "Group") +
  ggtitle("Lempel-Ziv Complexity") +
  coord_cartesian(ylim = c(0.58, 0.67)) +
  geom_signif(y_position = 0.667, xmin = 0.8, xmax = 1.2, annotation = "*", tip_length = 0.05, size = 1.4, textsize = 9, vjust = 0.4) +
  geom_signif(y_position = 0.62, xmin = 1.8, xmax = 2.2, annotation = "n.s.", tip_length = 0.05, size = 1.4, textsize = 9, vjust = -0.15)

print(p_LZ)
