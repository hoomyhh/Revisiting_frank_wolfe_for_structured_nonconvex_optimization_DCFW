clc; clear; close all;

%% Parameters
embedding_dim = 300;  % Dimension of embeddings
num_words = 10000;    % Number of words to load (adjust as needed)

%% File paths (update according to your setup)
% First, create a folder called "data" and download the datasets from:
% https://fasttext.cc/docs/en/english-vectors.html
% https://fasttext.cc/docs/en/crawl-vectors.html

file_en = 'data/wiki-news-300d-1M.vec';  % English embeddings
file_fr = 'data/cc.fr.300.vec';          % French embeddings

%% Load English Embeddings
fprintf('Loading English embeddings...\n');
words_en = strings(num_words, 1);
E1 = zeros(embedding_dim, num_words);

fid1 = fopen(file_en, 'r');
first_line = fgetl(fid1);  % Read the first line to check metadata
if contains(first_line, ' ')
    % Check if first line is metadata (number of words and dimensions)
    if all(isstrprop(first_line(1:5), 'digit'))
        % If the first line starts with numbers, it's metadata, so skip it
        line = fgetl(fid1);
    else
        % If not, treat the first line as an actual word embedding
        line = first_line;
    end
else
    line = first_line;
end

for i = 1:num_words
    if ~ischar(line)
        break;
    end
    data = strsplit(line, ' ');
    words_en(i) = string(data{1});
    E1(:, i) = str2double(data(2:end));
    line = fgetl(fid1);  % Read next line
end
fclose(fid1);
fprintf('English embeddings loaded.\n');

%% Load French Embeddings
fprintf('Loading French embeddings...\n');
words_fr = strings(num_words, 1);
E2 = zeros(embedding_dim, num_words);

fid2 = fopen(file_fr, 'r');
first_line = fgetl(fid2);  % Read the first line to check metadata
if contains(first_line, ' ')
    % Check if first line is metadata (number of words and dimensions)
    if all(isstrprop(first_line(1:5), 'digit'))
        % If the first line starts with numbers, it's metadata, so skip it
        line = fgetl(fid2);
    else
        % If not, treat the first line as an actual word embedding
        line = first_line;
    end
else
    line = first_line;
end

for i = 1:num_words
    if ~ischar(line)
        break;
    end
    data = strsplit(line, ' ');
    words_fr(i) = string(data{1});
    E2(:, i) = str2double(data(2:end));
    line = fgetl(fid2);  % Read next line
end
fclose(fid2);
fprintf('French embeddings loaded.\n');

% Save loaded embeddings for future use
save('fasttext_embeddings.mat', 'words_en', 'E1', 'words_fr', 'E2');
fprintf('Embeddings loaded and saved as fasttext_embeddings.mat\n');