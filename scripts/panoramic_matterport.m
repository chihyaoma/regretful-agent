add_path;

intrinsics = [1280 1024 1072.27 1073.08 626.949 506.267 0 0 0 0 0];

%% First we stitch 6 horizontal views from undistorted_color_images.
% Note that the images may under different exposure and thus you see color
% changes across image boundaries. Also we assume 6 views are from the same
% camera center and adjacent views have 60 degree between them, which may
% not be true.
% sepImg = [];
% vx = [-pi -2/3*pi -1/3*pi 0 1/3*pi 2/3*pi];
% vy = [0 0 0 0 0 0];
% for a = 1:6
%     img = im2double(imread(sprintf('./data/matterport3d_example/6f4d197078d14d6e944ed6533598a6f9_i1_%d.jpg', a-1)));
%     img = img(1:1013,1:1254,:); % cut the image such that camera center is at image center
%     sepImg(a).img = img;
%     sepImg(a).vx = vx(a);
%     sepImg(a).vy = 0;
%     sepImg(a).fov = 1.06; % calculate the fov with focal length and image size
%     sepImg(a).sz = size(sepImg(a).img);
% end
% 
% panocolor = combineViews( sepImg, 2048, 1024 );
% figure; imshow(panocolor);
% 
% %% Second we stitch 6 horizontal views from undistorted_depth_images.
% % Again, we do the same assumption for camera center and camera viewpoints.
% % Note that this depth is pixel-wise aligned with the color image above.
% sepImg = [];
% for a = 1:6
%     img = im2double(imread(sprintf('./data/matterport3d_example/6f4d197078d14d6e944ed6533598a6f9_d1_%d.png', a-1)));
%     img = img(1:1013,1:1254,:);
%     sepImg(a).img = img;
%     sepImg(a).vx = vx(a);
%     sepImg(a).vy = vy(a);
%     sepImg(a).fov = 1.06;
%     sepImg(a).sz = size(sepImg(a).img);
% end
% 
% panodepth = combineViews( sepImg, 2048, 1024 );
% figure; imshow(panodepth,[]);


%% This code can also stitch skybox images to panorama, which looks perfect
sepImg = [];

vx = [-pi/2 -pi/2 0 pi/2 pi -pi/2];
vy = [pi/2 0 0 0 0 -pi/2];

for a = 1:6
    sepImg(a).img = im2double(imread(sprintf('./data/matterport3d_skybox/6f4d197078d14d6e944ed6533598a6f9_skybox%d_sami.jpg',a-1)));
    sepImg(a).vx = vx(a);
    sepImg(a).vy = vy(a);
    sepImg(a).fov = pi/2+0.001;
    sepImg(a).sz = size(sepImg(a).img);
end

panoskybox = combineViews( sepImg, 2048, 1024 );
figure; imshow(panoskybox);


