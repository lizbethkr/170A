USE cs_project;

-- 1. csrankings.csv
DROP TABLE IF EXISTS csrankings;
CREATE TABLE csrankings (
                            id INT AUTO_INCREMENT PRIMARY KEY,
                            name VARCHAR(255),
                            first_name VARCHAR(255),
                            middle_name VARCHAR(255),
                            last_name VARCHAR(255),
                            affiliation VARCHAR(255),
                            homepage VARCHAR(255),
                            scholarid VARCHAR(50)
);

-- 2. generated-author-info.csv
DROP TABLE IF EXISTS generated_author_info;
CREATE TABLE generated_author_info (
                                       id INT AUTO_INCREMENT PRIMARY KEY,
                                       name VARCHAR(255),
                                       first_name VARCHAR(255),
                                       middle_name VARCHAR(255),
                                       last_name VARCHAR(255),
                                       dept VARCHAR(255),
                                       area VARCHAR(50),
                                       count INT,
                                       adjustedcount FLOAT,
                                       year INT
);

-- 3. acm-fellows.csv
DROP TABLE IF EXISTS acm_fellows;
CREATE TABLE acm_fellows (
                             id INT AUTO_INCREMENT PRIMARY KEY,
                             name VARCHAR(255),
                             year INT
);

-- 4. turing.csv
DROP TABLE IF EXISTS turing_awards;
CREATE TABLE turing_awards (
                               id INT AUTO_INCREMENT PRIMARY KEY,
                               name VARCHAR(255),
                               year INT
);

-- 5. geolocation.csv
DROP TABLE IF EXISTS geolocation;
CREATE TABLE geolocation (
                             id INT AUTO_INCREMENT PRIMARY KEY,
                             institution VARCHAR(255),
                             latitude DECIMAL(9,6),
                             longitude DECIMAL(9,6)
);

-- 6. country-info.csv
DROP TABLE IF EXISTS country_info;
CREATE TABLE country_info (
                              id INT AUTO_INCREMENT PRIMARY KEY,
                              institution VARCHAR(255),
                              region VARCHAR(100),
                              countryabbrv VARCHAR(10)
);

-- 7. dblp-aliases.csv
DROP TABLE IF EXISTS dblp_aliases;
CREATE TABLE dblp_aliases (
                              id INT AUTO_INCREMENT PRIMARY KEY,
                              alias VARCHAR(255),
                              name VARCHAR(255)
);

-- 8. conference_ranking.csv
DROP TABLE IF EXISTS conference_ranking;
CREATE TABLE conference_ranking (
                                    id INT AUTO_INCREMENT PRIMARY KEY,
                                    acronym VARCHAR(50),
                                    full_name VARCHAR(255),
                                    conference_rank VARCHAR(10),
                                    academic_society VARCHAR(20)
);

-- 9. US News data.csv
DROP TABLE IF EXISTS usnews_university_rankings;
CREATE TABLE usnews_university_rankings (
                                            id INT AUTO_INCREMENT PRIMARY KEY,
                                            institution_displayName VARCHAR(255),
                                            institution_schoolType VARCHAR(50),
                                            institution_aliasNames VARCHAR(255),
                                            institution_state VARCHAR(50),
                                            institution_city VARCHAR(50),
                                            institution_zip VARCHAR(20),
                                            institution_region VARCHAR(50),
                                            institution_isPublic BOOLEAN,
                                            institution_institutionalControl VARCHAR(50),
                                            institution_primaryPhotoCardThumb VARCHAR(255),
                                            ranking_displayRank VARCHAR(50),
                                            ranking_sortRank INT,
                                            ranking_isTied BOOLEAN,
                                            searchData_actAvg_rawValue FLOAT,
                                            searchData_percentReceivingAid_rawValue FLOAT,
                                            searchData_acceptanceRate_rawValue FLOAT,
                                            searchData_tuition_rawValue FLOAT,
                                            searchData_hsGpaAvg_rawValue FLOAT,
                                            searchData_engineeringRepScore_rawValue FLOAT,
                                            searchData_parentRank_rawValue FLOAT,
                                            searchData_enrollment_rawValue INT,
                                            searchData_businessRepScore_rawValue FLOAT,
                                            searchData_satAvg_rawValue FLOAT,
                                            searchData_costAfterAid_rawValue FLOAT,
                                            searchData_testAvgs_displayValue_0_value VARCHAR(50),
                                            searchData_testAvgs_displayValue_1_value VARCHAR(50)
);

-- 10. field_conference.csv
DROP TABLE IF EXISTS field_conference;
CREATE TABLE field_conference (
                                  id INT AUTO_INCREMENT PRIMARY KEY,
                                  major VARCHAR(50),
                                  field VARCHAR(100),
                                  conference VARCHAR(50)
);

SHOW TABLES;
